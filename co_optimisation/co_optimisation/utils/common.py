"""Structures shared by the design generators, the design filters and the runner.

Population moved here from runners/usd_generator.py, which still imports it, so that
``from co_optimisation.runners.usd_generator import Population`` keeps working.
"""

from __future__ import annotations


class Population:
    """A fixed set of robot designs, one per individual.

    Each individual is a Universal Scene Description path, an actuator
    parameter dict overriding the Python-side ``IdentifiedActuator``
    attributes, and an absolute link-geometry dict for in-place morphology
    updates.  ``root_body`` is the name of the emitted document's root body,
    which the Isaac Sim MuJoCo importer interposes between the referenced
    asset prim and every link prim, so ``utils/update.py`` needs it to address
    a link.
    """

    def __init__(
        self,
        usd_files: list[str],
        actuator_params: list[dict[str, dict]],
        link_length_params: list[dict[str, dict]],
        root_body: str,
        specs: list | None = None,
        nominal: list[dict] | None = None,
        parameters: list[dict] | None = None,
        xml_files: list[str] | None = None,
    ) -> None:
        self._usd_files = usd_files
        self._actuator_params = actuator_params
        self._link_length_params = link_length_params
        self._root_body = root_body
        self._specs = specs if specs is not None else []
        self._nominal = nominal
        self._parameters = parameters if parameters is not None else []
        self._xml_files = xml_files if xml_files is not None else []

    def get_usd_files(self) -> list[str]:
        """Return a list of Universal Scene Description paths, one per individual."""
        return self._usd_files

    def get_actuator_params(self) -> list[dict[str, dict]]:
        """Return actuator override dicts, one per individual.

        Each element is keyed by actuator group name (``"abad"``, ``"hip"``,
        ``"knee"``, ``"ankle"``), whose value is a dict of scalar overrides for
        the ``IdentifiedActuator`` attributes, as
        ``design_generator.common.actuator_catalog.SIM_PARAM_KEYS`` names them.
        """
        return self._actuator_params

    def get_link_length_params(self) -> list[dict[str, dict]]:
        """Return absolute link-geometry dicts, one per individual."""
        return self._link_length_params

    def get_root_body(self) -> str:
        """Return the emitted document's root body name."""
        return self._root_body

    def get_specs(self) -> list:
        """Return the ``DesignSpec`` behind each individual, for logging."""
        return self._specs

    def get_nominal(self) -> list[dict] | None:
        """Return each individual's nominal pose record.

        A record carries ``offsets`` (thigh, knee and ankle, radians about +y,
        the quadruped's ankle 0.0), ``dx`` (foot ahead of the hip pitch axis),
        ``depth`` (foot body below the root), ``sole`` (sole below the foot
        body) and ``height`` (``depth + sole``, the standing height), all from
        Family.chain on the emitted specification.
        """
        return self._nominal

    def get_parameters(self) -> list[dict]:
        """Return the searched parameters of each individual and whether its design emitted, for logging."""
        return self._parameters

    def get_xml_files(self) -> list[str]:
        """Return the MJCF document emitted for each individual."""
        return self._xml_files

    def set_individual(self, k: int, row: dict) -> None:
        """Replace the records at position *k* with *row*, keyed as DesignGeneratorBase._individual returns them."""
        self._usd_files[k] = row["usd_file"]
        self._actuator_params[k] = row["actuator_params"]
        self._link_length_params[k] = row["link_length_params"]
        self._specs[k] = row["spec"]
        self._nominal[k] = row["nominal"]
        self._parameters[k] = row["parameters"]
        self._xml_files[k] = row["xml_file"]
