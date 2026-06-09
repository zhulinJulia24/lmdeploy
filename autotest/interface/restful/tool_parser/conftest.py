import pytest
from utils.interface_utils import parametrize_interface
from utils.tool_reasoning_definitions import make_logged_client, setup_log_file


def tool_suite_marks(cls):
    """Config-driven ``tool_parser`` suite with shared API marks."""
    cls = pytest.mark.order(8)(cls)
    cls = pytest.mark.tool_call(cls)
    cls = pytest.mark.flaky(reruns=2)(cls)
    return parametrize_interface('tool_parser')(cls)


# ---------------------------------------------------------------------------
# Logging helpers – uses shared StreamTee / setup_log_file / make_logged_client
# from utils.tool_reasoning_definitions.
# ---------------------------------------------------------------------------


class _ToolCallTestBase:
    """Mixin providing per-test API request/response logging to *log_path*."""

    @pytest.fixture(autouse=True)
    def _setup_logging(self, request, config, backend, model_case):
        """Create the log directory and compute the log-file path."""
        self._log_file = setup_log_file(config, request.node.name, 'tool_calls')

    def _get_client(self):
        """Return *(client, model_name)* with transparent logging."""
        base_url = getattr(self, 'BASE_URL', None)
        return make_logged_client(self._log_file, base_url=base_url)


# ---------------------------------------------------------------------------
# Message constants
# ---------------------------------------------------------------------------

MESSAGES_ASKING_FOR_WEATHER = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant that can use tools. '
        'When asked about weather, use the get_current_weather tool.',
    },
    {
        'role': 'user',
        'content': "What's the weather like in Dallas, TX?",
    },
]

MESSAGES_ASKING_FOR_SEARCH = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to tools. '
        'Use the web_search tool when asked to look something up.',
    },
    {
        'role': 'user',
        'content': 'Search the web for the latest news about AI.',
    },
]

MESSAGES_ASKING_FOR_CALCULATION = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant. When asked math questions, '
        'use the calculate tool.',
    },
    {
        'role': 'user',
        'content': 'What is 1234 * 5678?',
    },
]

MESSAGES_ASKING_FOR_WEATHER_CN = [
    {
        'role': 'system',
        'content': '你是一个有用的助手，可以使用工具。'
        '当被问到天气时，请使用get_current_weather工具。',
    },
    {
        'role': 'user',
        'content': '北京今天的天气怎么样？',
    },
]

MESSAGES_NO_TOOL_NEEDED = [
    {
        'role': 'user',
        'content': 'Hi, please introduce yourself briefly.',
    },
]

MESSAGES_PARALLEL_WEATHER = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant. When asked about weather '
        'in multiple cities, call the weather tool for each city '
        'separately.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX and also in "
        'San Francisco, CA?',
    },
]

MESSAGES_PARALLEL_MIXED = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to multiple tools. '
        'You can call multiple tools in parallel when needed.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
        'Also calculate 1234 * 5678.',
    },
]
