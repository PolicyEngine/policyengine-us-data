"""H5 file writing and basic verification for local H5 publishing."""

from __future__ import annotations

from pathlib import Path

from .variables import H5Payload


class H5Writer:
    """Persist local-H5 payloads and provide lightweight output verification."""

    def write_payload(self, payload: H5Payload, output_path: str | Path) -> Path:
        import h5py

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "w") as h5_file:
            for variable, periods in payload.variables.items():
                group = h5_file.create_group(variable)
                for period, values in periods.items():
                    group.create_dataset(str(period), data=values)

        return output_path

    def verify_output(
        self,
        output_path: str | Path,
        *,
        time_period: int | str,
    ) -> dict[str, int | float]:
        import h5py

        output_path = Path(output_path)
        period = str(time_period)
        summary: dict[str, int | float] = {}

        with h5py.File(str(output_path), "r") as h5_file:
            household_dataset = self._get_dataset(h5_file, "household_id", period)
            if household_dataset is not None:
                summary["household_count"] = int(len(household_dataset[:]))

            person_dataset = self._get_dataset(h5_file, "person_id", period)
            if person_dataset is not None:
                summary["person_count"] = int(len(person_dataset[:]))

            household_weight_dataset = self._get_dataset(
                h5_file,
                "household_weight",
                period,
            )
            if household_weight_dataset is not None:
                summary["household_weight_sum"] = float(
                    household_weight_dataset[:].sum()
                )

            person_weight_dataset = self._get_dataset(
                h5_file,
                "person_weight",
                period,
            )
            if person_weight_dataset is not None:
                summary["person_weight_sum"] = float(person_weight_dataset[:].sum())

        return summary

    def _get_dataset(self, h5_file, variable: str, period: str):
        if variable not in h5_file:
            return None
        group = h5_file[variable]
        if period not in group:
            return None
        return group[period]
