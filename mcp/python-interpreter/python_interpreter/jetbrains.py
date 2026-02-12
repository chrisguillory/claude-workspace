"""JetBrains IDE interpreter and run configuration discovery.

Discovers Python interpreters and 'Run with Python Console' configurations
from JetBrains IDEs (PyCharm, IntelliJ IDEA, etc.) by parsing:
- jdk.table.xml for registered Python SDK entries
- *.run.xml files for shared run configurations

SDK resolution chain for run configs with empty SDK_HOME:
    .idea/misc.xml → project-jdk-name → jdk.table.xml → homePath
"""

from __future__ import annotations

__all__ = [
    'discover_sdk_entries',
    'discover_run_configs',
]

import os
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence

from python_interpreter.models import JetBrainsRunConfig, JetBrainsSDKEntry

_HOME = pathlib.Path.home()
_USER_HOME_VAR = '$USER_HOME$'
_PROJECT_DIR_VAR = '$PROJECT_DIR$'


def discover_sdk_entries(project_dir: pathlib.Path) -> Sequence[JetBrainsSDKEntry]:
    """Discover Python SDK entries associated with the project from jdk.table.xml.

    Scans all JetBrains IDE config directories for Python SDK entries whose
    ASSOCIATED_PROJECT_PATH matches project_dir or any subdirectory (monorepo support).
    Only returns entries where the Python executable exists on disk.
    """
    project_str = str(project_dir)
    all_entries: list[JetBrainsSDKEntry] = []

    for config_dir in _find_jetbrains_config_dirs():
        jdk_table = config_dir / 'options' / 'jdk.table.xml'
        if not jdk_table.exists():
            continue

        for entry in _parse_jdk_table(jdk_table):
            # Filter: associated with this project or a subdirectory
            if entry.associated_project is None:
                continue
            if not entry.associated_project.startswith(project_str):
                continue
            # Filter: executable exists
            if not pathlib.Path(entry.python_path).exists():
                continue
            all_entries.append(entry)

    # Deduplicate by name (latest JetBrains version wins — dirs are sorted)
    seen: dict[str, JetBrainsSDKEntry] = {}
    for entry in all_entries:
        seen[entry.name] = entry
    return list(seen.values())


def discover_run_configs(
    project_dir: pathlib.Path,
    sdk_entries: Sequence[JetBrainsSDKEntry] | None = None,
) -> Sequence[JetBrainsRunConfig]:
    """Discover 'Run with Python Console' configurations from .run.xml files.

    Scans for *.run.xml files anywhere in the project tree (excluding .idea/)
    and .idea/runConfigurations/*.xml. Only includes PythonConfigurationType
    configs where SHOW_COMMAND_LINE=true.

    Uses sdk_entries for resolving empty SDK_HOME via the misc.xml → jdk.table.xml chain.
    """
    xml_files = _scan_run_xml_files(project_dir)

    # Build SDK lookup from provided entries or discover fresh
    if sdk_entries is None:
        sdk_entries = discover_sdk_entries(project_dir)
    sdk_by_name = {e.name: e for e in sdk_entries}

    # Also parse all jdk.table entries (not just project-filtered) for SDK_NAME resolution
    all_sdk_by_name: dict[str, JetBrainsSDKEntry] = dict(sdk_by_name)
    for config_dir in _find_jetbrains_config_dirs():
        jdk_table = config_dir / 'options' / 'jdk.table.xml'
        if jdk_table.exists():
            for entry in _parse_jdk_table(jdk_table):
                all_sdk_by_name.setdefault(entry.name, entry)

    configs: list[JetBrainsRunConfig] = []
    for xml_path in xml_files:
        config = _parse_run_xml(xml_path, all_sdk_by_name)
        if config is not None:
            configs.append(config)

    return configs


# ---------------------------------------------------------------------------
# SDK table parsing
# ---------------------------------------------------------------------------


def _find_jetbrains_config_dirs() -> Sequence[pathlib.Path]:
    """Find all JetBrains IDE config directories, sorted by version (latest last)."""
    base = _HOME / 'Library' / 'Application Support' / 'JetBrains'
    if not base.exists():
        return []

    dirs = [
        child
        for child in base.iterdir()
        if child.is_dir() and (child.name.startswith('PyCharm') or child.name.startswith('IntelliJIdea'))
    ]
    return sorted(dirs, key=lambda p: p.name)


def _parse_jdk_table(jdk_table_path: pathlib.Path) -> Sequence[JetBrainsSDKEntry]:
    """Parse jdk.table.xml into JetBrainsSDKEntry list."""
    try:
        tree = ET.parse(jdk_table_path)  # noqa: S314
    except ET.ParseError:
        return []

    entries: list[JetBrainsSDKEntry] = []
    for jdk in tree.getroot().iter('jdk'):
        type_el = jdk.find('type')
        if type_el is None or type_el.get('value') != 'Python SDK':
            continue

        name_el = jdk.find('name')
        home_el = jdk.find('homePath')
        ver_el = jdk.find('version')
        additional = jdk.find('additional')

        if name_el is None or home_el is None:
            continue

        name = name_el.get('value', '')
        python_path = _resolve_user_home(home_el.get('value', ''))
        version = ver_el.get('value') if ver_el is not None else None

        flavor: str | None = None
        associated_project: str | None = None
        if additional is not None:
            raw_project = additional.get('ASSOCIATED_PROJECT_PATH')
            if raw_project:
                associated_project = _resolve_user_home(raw_project)
            for setting in additional.findall('setting'):
                if setting.get('name') == 'FLAVOR_ID':
                    flavor = setting.get('value')

        entries.append(
            JetBrainsSDKEntry(
                name=name,
                python_path=python_path,
                version=version,
                flavor=flavor,
                associated_project=associated_project,
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Run configuration parsing
# ---------------------------------------------------------------------------


_SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'dist', 'build', '.tox', '.mypy_cache'}


def _scan_run_xml_files(project_dir: pathlib.Path) -> Iterator[pathlib.Path]:
    """Find all candidate run configuration XML files in the project tree."""
    # *.run.xml anywhere in project tree (excluding .idea/ and heavy dirs)
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and d != '.idea']
        for f in files:
            if f.endswith('.run.xml'):
                yield pathlib.Path(root) / f

    # .idea/runConfigurations/*.xml (legacy location)
    run_configs_dir = project_dir / '.idea' / 'runConfigurations'
    if run_configs_dir.is_dir():
        yield from run_configs_dir.glob('*.xml')

    # Also check subdirectories for monorepo support
    for child in project_dir.iterdir():
        if not child.is_dir() or child.name.startswith('.'):
            continue
        sub_run_configs = child / '.idea' / 'runConfigurations'
        if sub_run_configs.is_dir():
            yield from sub_run_configs.glob('*.xml')


def _parse_run_xml(
    xml_path: pathlib.Path,
    sdk_lookup: dict[str, JetBrainsSDKEntry],
) -> JetBrainsRunConfig | None:
    """Parse a single run configuration XML. Returns None if not a Python console config."""
    try:
        tree = ET.parse(xml_path)  # noqa: S314
    except ET.ParseError:
        return None

    root = tree.getroot()

    # Find the configuration element (may be wrapped in <component>)
    config_el = root.find('.//configuration')
    if config_el is None:
        return None

    # Must be PythonConfigurationType
    if config_el.get('type') != 'PythonConfigurationType':
        return None

    # Must have SHOW_COMMAND_LINE=true ("Run with Python Console")
    if not _get_option_bool(config_el, 'SHOW_COMMAND_LINE'):
        return None

    name = config_el.get('name', xml_path.stem)

    # Find the project dir for $PROJECT_DIR$ resolution
    idea_dir = _find_idea_dir(xml_path)
    config_project_dir = idea_dir.parent if idea_dir else xml_path.parent

    # Extract fields
    sdk_home = _resolve_variables(_get_option(config_el, 'SDK_HOME') or '', config_project_dir)
    sdk_name = _get_option(config_el, 'SDK_NAME')
    working_dir = _resolve_variables(_get_option(config_el, 'WORKING_DIRECTORY') or '', config_project_dir)
    script_name = _resolve_variables(_get_option(config_el, 'SCRIPT_NAME') or '', config_project_dir)
    parameters = _get_option(config_el, 'PARAMETERS')
    env_vars = _extract_env_vars(config_el)

    # Resolve python path
    python_path: str | None = sdk_home if sdk_home else None

    if not python_path:
        # Try SDK_NAME lookup in jdk.table
        if sdk_name and sdk_name in sdk_lookup:
            python_path = sdk_lookup[sdk_name].python_path
        else:
            # Try misc.xml → project-jdk-name → jdk.table
            python_path = _resolve_sdk_via_misc_xml(idea_dir, sdk_lookup)

    return JetBrainsRunConfig(
        name=name,
        xml_path=str(xml_path),
        python_path=python_path,
        cwd=working_dir or None,
        env=env_vars,
        script_name=script_name or None,
        parameters=parameters or None,
    )


# ---------------------------------------------------------------------------
# SDK resolution helpers
# ---------------------------------------------------------------------------


def _resolve_sdk_via_misc_xml(
    idea_dir: pathlib.Path | None,
    sdk_lookup: dict[str, JetBrainsSDKEntry],
) -> str | None:
    """Resolve Python path via .idea/misc.xml → project-jdk-name → jdk.table.xml."""
    if idea_dir is None:
        return None

    jdk_name = _get_jdk_name_from_misc_xml(idea_dir)
    if jdk_name is None:
        return None

    entry = sdk_lookup.get(jdk_name)
    return entry.python_path if entry else None


def _get_jdk_name_from_misc_xml(idea_dir: pathlib.Path) -> str | None:
    """Read project-jdk-name from .idea/misc.xml."""
    misc_xml = idea_dir / 'misc.xml'
    if not misc_xml.exists():
        return None

    try:
        tree = ET.parse(misc_xml)  # noqa: S314
    except ET.ParseError:
        return None

    for component in tree.getroot().iter('component'):
        if component.get('name') == 'ProjectRootManager':
            return component.get('project-jdk-name')

    return None


def _find_idea_dir(starting_from: pathlib.Path) -> pathlib.Path | None:
    """Walk up from a file path to find the nearest .idea/ directory."""
    current = starting_from.parent if starting_from.is_file() else starting_from
    for _ in range(10):  # depth limit
        idea = current / '.idea'
        if idea.is_dir():
            return idea
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# XML utility helpers
# ---------------------------------------------------------------------------


def _get_option(element: ET.Element, name: str) -> str | None:
    """Get option value from <option name="..." value="..."> element."""
    for option in element.findall('option'):
        if option.get('name') == name:
            return option.get('value')
    return None


def _get_option_bool(element: ET.Element, name: str) -> bool:
    """Get boolean option value. Returns False if not found."""
    value = _get_option(element, name)
    return value is not None and value.lower() == 'true'


def _extract_env_vars(config_element: ET.Element) -> dict[str, str] | None:
    """Extract environment variables from <envs> element."""
    envs_el = config_element.find('envs')
    if envs_el is None:
        return None

    env_dict: dict[str, str] = {}
    for env in envs_el.findall('env'):
        name = env.get('name')
        value = env.get('value')
        if name is not None and value is not None:
            env_dict[name] = value

    return env_dict if env_dict else None


def _resolve_variables(value: str, project_dir: pathlib.Path) -> str:
    """Replace JetBrains path variables with actual values."""
    value = value.replace(_PROJECT_DIR_VAR, str(project_dir))
    value = value.replace(_USER_HOME_VAR, str(_HOME))
    return value


def _resolve_user_home(value: str) -> str:
    """Replace $USER_HOME$ with actual home directory."""
    return value.replace(_USER_HOME_VAR, str(_HOME))
