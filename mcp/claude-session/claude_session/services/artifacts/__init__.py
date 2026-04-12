"""
Artifact handling modules for session clone/archive/restore/delete operations.

This package contains modules for collecting, transforming, and writing
session-specific artifacts like tool results, todos, plan files, etc.
"""

from __future__ import annotations

# ToolResultExtension lives in claude_session/schemas/types (avoids circular import)
from claude_session.schemas.types import ToolResultExtension

from .agent_ids import (
    AGENT_FILENAME_PATTERN,
    AgentFileInfo,
    apply_agent_id_mapping,
    collect_agent_file_info,
    detect_agent_structure,
    extract_agent_ids_from_files,
    extract_base_agent_id,
    generate_agent_id_mapping,
    generate_clone_agent_id,
    transform_agent_filename,
)
from .custom_title import (
    CLONE_SUFFIX_PATTERN,
    extract_base_custom_title,
    extract_custom_title_from_file,
    extract_custom_title_from_records,
    generate_clone_custom_title,
)
from .debug_log import (
    collect_debug_log,
    write_debug_log,
)
from .jsonl import write_jsonl
from .paths import (
    MissingCwdError,
    extract_source_project_path,
)
from .plan_files import (
    SLUG_RECORD_TYPES,
    apply_slug_mapping,
    collect_plan_files,
    extract_base_slug,
    extract_slugs_from_records,
    generate_clone_slug,
    write_plan_files,
)
from .session_env import (
    collect_session_env,
    create_session_env_dir,
    get_session_env_dir,
    write_session_env,
)
from .session_memory import (
    SESSION_MEMORY_DIRNAME,
    collect_session_memory,
    write_session_memory,
)
from .tasks import (
    TASK_METADATA_FILES,
    TaskDirectoryContents,
    classify_task_directory,
    collect_task_metadata,
    get_tasks_dir,
    iter_task_paths,
    iter_tasks,
    write_task_metadata,
    write_tasks,
)
from .todos import (
    collect_todos,
    get_todos_dir,
    transform_todo_filename,
    write_todos,
)
from .tool_results import (
    TOOL_RESULT_EXTENSIONS,
    DiscoveredDirectory,
    DiscoveredFile,
    DiscoveryResult,
    ToolResultCollection,
    ToolResultDirectory,
    ToolResultDirectoryFile,
    ToolResultFile,
    collect_tool_results,
    discover_tool_results,
    get_tool_results_dir,
    write_tool_results,
)

__all__ = [
    # agent_ids
    'AGENT_FILENAME_PATTERN',
    'AgentFileInfo',
    'apply_agent_id_mapping',
    'collect_agent_file_info',
    'detect_agent_structure',
    'extract_agent_ids_from_files',
    'extract_base_agent_id',
    'generate_agent_id_mapping',
    'generate_clone_agent_id',
    'transform_agent_filename',
    # custom_title
    'CLONE_SUFFIX_PATTERN',
    'extract_base_custom_title',
    'extract_custom_title_from_file',
    'extract_custom_title_from_records',
    'generate_clone_custom_title',
    # debug_log
    'collect_debug_log',
    'write_debug_log',
    # jsonl
    'write_jsonl',
    # paths
    'MissingCwdError',
    'extract_source_project_path',
    # plan_files
    'SLUG_RECORD_TYPES',
    'apply_slug_mapping',
    'collect_plan_files',
    'extract_base_slug',
    'extract_slugs_from_records',
    'generate_clone_slug',
    'write_plan_files',
    # session_env
    'collect_session_env',
    'create_session_env_dir',
    'get_session_env_dir',
    'write_session_env',
    # session_memory
    'SESSION_MEMORY_DIRNAME',
    'collect_session_memory',
    'write_session_memory',
    # tasks
    'TASK_METADATA_FILES',
    'TaskDirectoryContents',
    'classify_task_directory',
    'collect_task_metadata',
    'get_tasks_dir',
    'iter_task_paths',
    'iter_tasks',
    'write_task_metadata',
    'write_tasks',
    # todos
    'collect_todos',
    'get_todos_dir',
    'transform_todo_filename',
    'write_todos',
    # tool_results
    'DiscoveredDirectory',
    'DiscoveredFile',
    'DiscoveryResult',
    'TOOL_RESULT_EXTENSIONS',
    'ToolResultCollection',
    'ToolResultDirectory',
    'ToolResultDirectoryFile',
    'ToolResultExtension',
    'ToolResultFile',
    'collect_tool_results',
    'discover_tool_results',
    'get_tool_results_dir',
    'write_tool_results',
]
