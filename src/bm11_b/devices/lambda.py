"""
Ophyd Async implementation for Lambad detector.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Sequence
from logging import getLogger
from pathlib import Path
from typing import Annotated as A
from typing import Any, cast
from urllib.parse import urlunparse

import numpy as np  # type: ignore[import-not-found]
from bluesky.protocols import StreamAsset
from event_model import (  # type: ignore[import-untyped]
    ComposeStreamResource,
    ComposeStreamResourceBundle,
    DataKey,  # type: ignore[import-untyped]
    StreamDatum,
    StreamRange,
    StreamResource,
)
from ophyd_async.core import (
    DetectorTrigger,
    HDFDatasetDescription,
    PathInfo,
    PathProvider,
    SignalDatatypeT,
    SignalR,
    SignalRW,
    StrictEnum,
    TriggerInfo,
    observe_value,
)
from ophyd_async.epics.adcore import (
    ADBaseController,
    ADBaseDatasetDescriber,
    ADBaseIO,
    ADImageMode,
    ADWriter,
    AreaDetector,
    NDFileHDFIO,
    NDPluginBaseIO,
)
from ophyd_async.epics.signal import PvSuffix

logger = getLogger(__name__)


class LambdaDocumentComposer:
    def __init__(
        self,
        full_file_name: Path,
        datasets: list[HDFDatasetDescription],
        last_emitted_index: int = 0,
        hostname: str = "localhost",
    ) -> None:
        self._last_emitted = last_emitted_index
        self._hostname = hostname
        uri = urlunparse(
            (
                "file",
                self._hostname,
                str(full_file_name.absolute()),
                "",
                "",
                None,
            )
        )
        bundler_composer = ComposeStreamResource()
        self._bundles: list[ComposeStreamResourceBundle] = [
            bundler_composer(
                mimetype="application/x-hdf5",
                uri=uri,
                data_key=ds.data_key,
                parameters={
                    "dataset": ds.dataset,
                    "chunk_shape": ds.chunk_shape,
                },
                uid=None,
                validate=True,
            )
            for ds in datasets
        ]

    def stream_resources(self) -> Iterator[StreamResource]:
        for bundle in self._bundles:
            yield bundle.stream_resource_doc

    def stream_data(self, indices_written: int) -> Iterator[StreamDatum]:
        if indices_written > self._last_emitted:
            indices: StreamRange = {
                "start": self._last_emitted,
                "stop": indices_written,
            }
            self._last_emitted = indices_written
            for bundle in self._bundles:
                yield bundle.compose_stream_datum(indices)


class LambdaSavingSatus(StrictEnum):
    """Header detail levels for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#stream-interface
    """

    Done = "Done"
    Writing = "Writing"


class LambdaDriverIO(ADBaseIO, NDFileHDF5IO):
    """Defines the full specifics of the Eiger driver.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#implementation-of-standard-driver-parameters
    """

    writefile: A[SignalR[LambdaSavingSatus], PvSuffix.rbv("WriteFile_RBV")]
    num_frames_flush: A[SignalRW[int], PvSuffix.rbv("NumFramesFlush")]
    ndadattribute_flush: A[SignalRW[int], PvSuffix.rbv("NDAttributeChunk")]


class LambdaWriter(ADWriter[LambdaDriverIO]):  # type: ignore[reportInvalidTypeArguments]
    """Eiger-specific file writer using the built-in FileWriter interface."""

    default_suffix: str = "HDF1:"

    def __init__(
        self,
        fileio: LambdaDriverIO,
        path_provider: PathProvider,
        dataset_describer: ADBaseDatasetDescriber,
        plugins: dict[str, NDPluginBaseIO] | None = None,
    ):
        super().__init__(
            fileio,
            path_provider,
            dataset_describer,
            file_extension=".h5",
            mimetype="application/x-hdf5",
            plugins=plugins,
        )

        self._file_info: PathInfo | None = None
        self._datasets: list[HDFDatasetDescription] = []
        self._master_file_path_cache: list[Path] = []

    async def open(self, name: str, exposures_per_event: int = 1) -> dict[str, DataKey]:
        """Setup file writing for acquisition."""
        # Get file path info from path provider
        if exposures_per_event != 1:
            raise ValueError("Only one exposure per event")
        # Cache for use later
        self._exposures_per_event = exposures_per_event

        # Force the number of images per file to a large number to simplify the logic
        detector_shape = await self._dataset_describer.shape()

        chunk_shape = cast(tuple[int, ...], (1, *detector_shape))
        frame_datasets = [
            HDFDatasetDescription(
                data_key="lambda_mosaic",
                dataset="entry/data/data",
                shape=(exposures_per_event, *detector_shape),
                # Always write as uint16
                dtype_numpy=np.dtype(np.uint16).str,
                chunk_shape=chunk_shape,
            )
        ]

        # Cache descriptions for later use
        self._datasets = frame_datasets

        return {
            ds.data_key: DataKey(
                source="ADLambda FileWriter",
                shape=list(ds.shape),
                dtype="array"
                if exposures_per_event > 1 or len(ds.shape) > 1
                else "number",
                dtype_numpy=ds.dtype_numpy,
                external="STREAM:",
            )
            for ds in self._datasets
        }

    async def collect_stream_docs(
        self, name: str, indices_written: int
    ) -> AsyncIterator[StreamAsset]:
        """Generate stream documents for the written HDF5 files."""
        if indices_written:
            master_file_path = await self._master_file_path
            if master_file_path is None:
                msg = f"Master file path is not set for {name}: {self._file_info}"
                raise ValueError(msg)

            # Eiger generates a new master file for each trigger
            # so we need to create a new composer with a new
            # master file path
            composer = LambdaDocumentComposer(
                master_file_path,
                self._datasets,
                last_emitted_index=indices_written - 1,
            )
            for doc in composer.stream_resources():
                yield "stream_resource", doc

            for doc in composer.stream_data(indices_written):
                yield "stream_datum", doc

    async def observe_indices_written(
        self, timeout: float
    ) -> AsyncGenerator[int, None]:
        async for num_captured in observe_value(self.fileio.array_counter, timeout):
            yield num_captured // self._exposures_per_event

    async def get_indices_written(self) -> int:
        return await self.fileio.array_counter.get_value() // self._exposures_per_event

    async def close(self) -> None:
        """Clean up file writing after acquisition and validate files exist."""

        # Check that the master files were written
        for master_file_path in self._master_file_path_cache:
            if not master_file_path.exists():
                logger.warning("Master file was not written: %s", master_file_path)

        self._file_info = None


class LambdaController(ADBaseController[LambdaDriverIO]):
    """Controller for Eiger detector, handling trigger modes and acquisition setup."""

    def __init__(
        self, driver: LambdaDriverIO, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(driver, *args, **kwargs)

    def get_deadtime(self, exposure: float | None) -> float:
        """Get detector deadtime for the given exposure."""
        default_deadtime = 0.000001
        if exposure is not None:
            logger.warning(
                "Ignoring exposure to calculate deadtime: %s, defaulting to %s",
                exposure,
                default_deadtime,
            )
        return default_deadtime

    async def prepare(self, trigger_info: TriggerInfo) -> None:
        """Prepare the detector for acquisition."""
        ...


class LmabdaDetector(AreaDetector[LambdaController]):
    """Eiger detector implementation using the AreaDetector pattern."""

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        driver_suffix: str = "cam1:",
        writer_cls: type[ADWriter] = LambdaWriter,  # type: ignore[reportUnknownParameterType]
        fileio_suffix: str | None = None,
        name: str = "",
        config_sigs: Sequence[SignalR[SignalDatatypeT]] = (),
        plugins: dict[str, NDPluginBaseIO] | None = None,
    ):
        driver = LambdaDriverIO(prefix + driver_suffix)
        controller = LambdaController(driver)
        if issubclass(writer_cls, LambdaWriter):
            dataset_describer = ADBaseDatasetDescriber(driver)
            # EigerWriter takes the driver as the fileio, since it relies on driver PVs
            writer = writer_cls(
                driver,
                path_provider,
                dataset_describer=dataset_describer,
                plugins=plugins,
            )
        else:
            writer = writer_cls.with_io(
                prefix,
                path_provider,
                dataset_source=driver,
                fileio_suffix=fileio_suffix,
                plugins=plugins,
            )

        super().__init__(
            controller=controller,
            writer=writer,
            plugins=plugins,
            name=name,
            config_sigs=config_sigs,
        )
