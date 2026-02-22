"""Tests for LibraryBoundary -- exception translation at library call boundaries.

Validates both explicit mode (context manager / decorator) and automatic mode
(proxy wrapping), including generator, async generator, and context manager
translation. Uses a realistic fake library module (fake_lib) to test proxy
wrapping the way real code wraps real libraries.
"""

from __future__ import annotations

import asyncio

import bashlex
import bashlex.errors
import pytest
from local_lib.library_boundary import LibraryBoundary

from tests.local_lib import fake_lib


class AppError(Exception):
    """Target exception for all tests."""


class AppErrorSubclass(AppError):
    """Subclass of target -- should also pass through the double-wrap guard."""


class NoArgError(Exception):
    """Target exception whose __init__ requires no arguments beyond self."""

    def __init__(self) -> None:
        super().__init__('fixed message')


class CustomInitError(Exception):
    """Target exception with a non-standard __init__ signature."""

    def __init__(self, msg: str, code: int = 0) -> None:
        super().__init__(msg)
        self.code = code


class TestContextManager:
    """Verify context manager exception translation."""

    def test_no_exception_passes_through(self) -> None:
        boundary = LibraryBoundary(AppError)
        with boundary:
            result = 1 + 1
        assert result == 2

    @pytest.mark.parametrize(
        'exception, message',
        [
            (ValueError, 'original'),
            (RuntimeError, 'the message'),
            (TypeError, 'type error'),
        ],
    )
    def test_translates_exception(self, exception: type[Exception], message: str) -> None:
        boundary = LibraryBoundary(AppError)
        with pytest.raises(AppError) as exc_info, boundary:
            raise exception(message)
        assert exc_info.value.args == (message,)
        assert isinstance(exc_info.value.__cause__, exception)

    def test_double_wrap_guard(self) -> None:
        """Exception already the target type passes through unchanged."""
        boundary = LibraryBoundary(AppError)
        original = AppError('already wrapped')
        with pytest.raises(AppError) as exc_info, boundary:
            raise original
        assert exc_info.value is original

    def test_double_wrap_guard_subclass(self) -> None:
        """Subclass of target type also passes through (isinstance check)."""
        boundary = LibraryBoundary(AppError)
        original = AppErrorSubclass('sub')
        with pytest.raises(AppErrorSubclass) as exc_info, boundary:
            raise original
        assert exc_info.value is original

    @pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit, GeneratorExit])
    def test_system_exception_passes_through(self, exception: type[BaseException]) -> None:
        """BaseException subclasses that aren't Exception pass through untranslated."""
        boundary = LibraryBoundary(AppError)
        with pytest.raises(exception) as exc_info, boundary:
            raise exception
        assert exc_info.value.args == ()

    @pytest.mark.parametrize('exception', [StopIteration, StopAsyncIteration])
    def test_control_flow_exception_passes_through(self, exception: type[Exception]) -> None:
        """Iterator protocol exceptions pass through untranslated (_PASSTHROUGH)."""
        boundary = LibraryBoundary(AppError)
        with pytest.raises(exception) as exc_info, boundary:
            raise exception
        assert exc_info.value.args == ()

    def test_preserves_traceback(self) -> None:
        """Original raise location is preserved as deepest traceback frame."""
        boundary = LibraryBoundary(AppError)

        def library_function() -> None:
            raise ValueError('deep')

        with pytest.raises(AppError) as exc_info, boundary:
            library_function()

        tb = exc_info.value.__traceback__
        assert tb is not None
        while tb.tb_next:
            tb = tb.tb_next
        assert tb.tb_frame.f_code.co_name == 'library_function'

    def test_exception_with_no_message(self) -> None:
        """ValueError() with no args produces empty string message."""
        boundary = LibraryBoundary(AppError)
        with pytest.raises(AppError) as exc_info, boundary:
            raise ValueError()
        assert exc_info.value.args == ('',)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_reentrant_boundary(self) -> None:
        """Same boundary instance used in nested with-blocks."""
        boundary = LibraryBoundary(AppError)
        with pytest.raises(AppError) as exc_info, boundary, boundary:
            raise ValueError('inner')
        assert exc_info.value.args == ('inner',)


class TestAsyncContextManager:
    """Verify async context manager behaves identically to sync."""

    async def test_translates_exception(self) -> None:
        boundary = LibraryBoundary(AppError)
        with pytest.raises(AppError) as exc_info:
            async with boundary:
                raise ValueError('async error')
        assert exc_info.value.args == ('async error',)

    async def test_no_exception_passes_through(self) -> None:
        boundary = LibraryBoundary(AppError)
        async with boundary:
            result = 42
        assert result == 42


class TestDecorator:
    """Verify decorator mode for sync and async functions."""

    def test_sync_decorator_translates(self) -> None:
        boundary = LibraryBoundary(AppError)

        @boundary
        def failing() -> None:
            raise ValueError('decorated')

        with pytest.raises(AppError) as exc_info:
            failing()
        assert exc_info.value.args == ('decorated',)

    def test_sync_decorator_returns_value(self) -> None:
        boundary = LibraryBoundary(AppError)

        @boundary
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_decorator_preserves_metadata(self) -> None:
        boundary = LibraryBoundary(AppError)

        @boundary
        def my_func() -> None:
            """My docstring."""

        @boundary
        async def my_async_func() -> None:
            """My async docstring."""

        assert my_func.__name__ == 'my_func'
        assert my_func.__doc__ == 'My docstring.'
        assert my_async_func.__name__ == 'my_async_func'
        assert my_async_func.__doc__ == 'My async docstring.'

    async def test_async_decorator_translates(self) -> None:
        boundary = LibraryBoundary(AppError)

        @boundary
        async def failing() -> None:
            raise ValueError('async decorated')

        with pytest.raises(AppError) as exc_info:
            await failing()
        assert exc_info.value.args == ('async decorated',)

    async def test_async_decorator_returns_value(self) -> None:
        boundary = LibraryBoundary(AppError)

        @boundary
        async def add(a: int, b: int) -> int:
            return a + b

        assert await add(3, 4) == 7

    async def test_async_auto_detected(self) -> None:
        """Decorator auto-detects async functions."""
        boundary = LibraryBoundary(AppError)

        @boundary
        async def coro() -> str:
            return 'hello'

        assert asyncio.iscoroutinefunction(coro)


class TestProxy:
    """Verify proxy wrapping for sync callables and attribute access on a module."""

    def test_wraps_module(self) -> None:
        """Wrapping a module -- the primary use case."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert proxy.greet('world') == 'Hello, world'

    def test_callable_translated(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            proxy.fail()
        assert exc_info.value.args == ('sync failure',)

    def test_callable_returns_value(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert proxy.add(3, 4) == 7

    def test_non_callable_passthrough(self) -> None:
        """Module-level attributes pass through unwrapped."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert proxy.version == '1.0'
        assert proxy.MAX_RETRIES == 3

    def test_missing_attribute_raises_attribute_error(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AttributeError) as exc_info:
            proxy.nonexistent
        assert exc_info.value.args == ("module 'tests.local_lib.fake_lib' has no attribute 'nonexistent'",)

    def test_hasattr_false_for_missing(self) -> None:
        """hasattr() returns False for nonexistent attributes (not AppError)."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert not hasattr(proxy, 'nonexistent')

    def test_wraps_preserves_metadata(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert proxy.greet.__name__ == 'greet'

    def test_double_wrap_guard(self) -> None:
        """Target exception raised by library passes through unchanged."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            proxy.fail_with_target(AppError)
        assert exc_info.value.args == ('already target',)
        assert exc_info.value.__cause__ is None

    def test_system_exception_passes_through(self) -> None:
        """KeyboardInterrupt is not translated even through proxy."""

        class _KBILib:
            def do(self) -> None:
                raise KeyboardInterrupt

        proxy = LibraryBoundary(AppError).wrap(_KBILib())
        with pytest.raises(KeyboardInterrupt) as exc_info:
            proxy.do()
        assert exc_info.value.args == ()

    def test_wrapping_class_instance(self) -> None:
        """Wrapping a class instance works the same as wrapping a module."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib.config)
        assert proxy.method() == 'method result'
        assert proxy.name == 'default'


class TestProxyCallableObjects:
    """Verify proxy wrapping of callable objects (instances with __call__)."""

    def test_callable_object_success(self) -> None:
        """Callable instance attribute is detected and wrapped."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert proxy.processor('data') == 'processed:data'

    def test_callable_object_failure_translated(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            proxy.failing_processor('data')
        assert exc_info.value.args == ('processing failed: data',)

    def test_callable_object_direct_wrap_not_callable(self) -> None:
        """Wrapping a callable object directly -- proxy itself is NOT callable.

        Python looks up __call__ on the type (_TranslatingProxy), not the
        instance. Since _TranslatingProxy has no __call__, the proxy can't
        be called directly. This is the same limitation as magic methods.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib.processor)
        with pytest.raises(TypeError) as exc_info:
            proxy('data')
        assert exc_info.value.args == ("'_TranslatingProxy' object is not callable",)


class TestProxyProperties:
    """Verify proxy behavior with @property attributes on wrapped objects."""

    def test_property_passthrough(self) -> None:
        """Properties on the wrapped object are accessed normally.

        The proxy's __getattr__ calls getattr() on the target, which triggers
        the property descriptor. The result is a plain value, not callable.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib.config)
        assert proxy.computed == 'computed:default'

    def test_failing_property_not_translated(self) -> None:
        """Property that raises -- the exception happens in getattr(), before
        the callable check. It propagates as-is because __getattr__ doesn't
        wrap the getattr() call in a boundary."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib.config)
        with pytest.raises(ValueError) as exc_info:
            proxy.failing_property
        assert exc_info.value.args == ('property access failed',)


class TestProxyMagicMethods:
    """Verify behavior of magic methods on proxied objects.

    Python looks up magic methods on the type, not the instance. This means
    __len__, __getitem__, etc. are NOT intercepted by __getattr__. The proxy
    does not implement these, so they raise TypeError.
    """

    def test_len_not_proxied(self) -> None:
        """len() on proxy raises TypeError -- magic methods bypass __getattr__."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib.data_store)
        with pytest.raises(TypeError) as exc_info:
            len(proxy)
        assert exc_info.value.args == ("object of type '_TranslatingProxy' has no len()",)

    def test_getitem_not_proxied(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib.data_store)
        with pytest.raises(TypeError) as exc_info:
            proxy[0]
        assert exc_info.value.args == ("'_TranslatingProxy' object is not subscriptable",)

    def test_contains_not_proxied(self) -> None:
        """'in' operator falls back to __iter__ then __getitem__, both missing."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib.data_store)
        with pytest.raises(TypeError) as exc_info:
            10 in proxy
        assert exc_info.value.args == ("argument of type '_TranslatingProxy' is not iterable",)

    def test_regular_method_still_works(self) -> None:
        """Regular methods on the same object work through __getattr__."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib.data_store)
        assert proxy.query() == [10, 20, 30]


class TestProxyNestedAccess:
    """Verify that nested attribute access returns unwrapped sub-objects."""

    def test_submodule_attribute_passthrough(self) -> None:
        """Sub-object returned by attribute access is NOT auto-wrapped."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        # fake_lib.sub is a non-callable object, returned as-is
        sub = proxy.sub
        assert sub.error_code == 42

    def test_submodule_method_not_translated(self) -> None:
        """Methods on sub-objects are NOT translated -- only the top-level proxy wraps."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        sub = proxy.sub
        # sub is the raw _SubModule instance, not a proxy
        with pytest.raises(ValueError) as exc_info:
            sub.fail()
        assert exc_info.value.args == ('submodule failure',)

    def test_submodule_can_be_wrapped_separately(self) -> None:
        """User can wrap sub-objects explicitly for recursive translation."""
        boundary = LibraryBoundary(AppError)
        proxy = boundary.wrap(fake_lib)
        sub_proxy = boundary.wrap(proxy.sub)
        with pytest.raises(AppError) as exc_info:
            sub_proxy.fail()
        assert exc_info.value.args == ('submodule failure',)


class TestProxyMultipleBoundaries:
    """Verify behavior when multiple boundaries wrap the same library."""

    def test_two_boundaries_same_library(self) -> None:
        """Two different boundaries can wrap the same library independently."""

        class ErrorA(Exception):
            pass

        class ErrorB(Exception):
            pass

        proxy_a = LibraryBoundary(ErrorA).wrap(fake_lib)
        proxy_b = LibraryBoundary(ErrorB).wrap(fake_lib)

        with pytest.raises(ErrorA) as exc_info_a:
            proxy_a.fail()
        assert exc_info_a.value.args == ('sync failure',)
        with pytest.raises(ErrorB) as exc_info_b:
            proxy_b.fail()
        assert exc_info_b.value.args == ('sync failure',)

    def test_recursive_proxy_double_wrap(self) -> None:
        """boundary.wrap(boundary.wrap(lib)) -- double-proxied.

        The outer proxy wraps calls. If the inner proxy translates first,
        the outer sees the target type and the double-wrap guard passes it
        through unchanged.
        """
        boundary = LibraryBoundary(AppError)
        inner = boundary.wrap(fake_lib)
        outer = boundary.wrap(inner)

        # Success still works
        assert outer.greet('world') == 'Hello, world'

        # Exception is translated once, not double-wrapped
        with pytest.raises(AppError) as exc_info:
            outer.fail()
        assert exc_info.value.args == ('sync failure',)
        assert isinstance(exc_info.value.__cause__, ValueError)
        # The cause chain should be ValueError, not AppError -> AppError
        assert not isinstance(exc_info.value.__cause__, AppError)


class TestProxyAsync:
    """Verify proxy wrapping for async callables."""

    async def test_async_callable_translated(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            await proxy.async_fail()
        assert exc_info.value.args == ('async failure',)

    async def test_async_callable_returns_value(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert await proxy.async_greet('world') == 'Hello async, world'


class TestProxyGenerator:
    """Verify generator wrapping translates exceptions during iteration."""

    def test_generator_translates_mid_iteration(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            list(proxy.items())
        assert exc_info.value.args == ('mid-iteration failure',)

    def test_generator_success(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        assert list(proxy.good_items()) == [10, 20, 30]

    def test_generator_send(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.echo_gen()
        assert next(gen) == 'ready'
        assert gen.send('hello') == 'echo:hello'
        assert gen.send('world') == 'echo:world'
        gen.close()

    def test_generator_close(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.items()
        next(gen)
        gen.close()  # should not raise

    def test_generator_throw_translates_response(self) -> None:
        """throw() forwards to generator; if generator raises differently, it's translated."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.echo_gen()
        next(gen)
        # Generator doesn't handle RuntimeError, so it propagates
        # through the boundary and gets translated
        with pytest.raises(AppError) as exc_info:
            gen.throw(RuntimeError('injected'))
        assert exc_info.value.args == ('injected',)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_generator_throw_converts_exception(self) -> None:
        """Generator catches thrown exception and raises a different one -- translated."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.throw_converter()
        next(gen)
        with pytest.raises(AppError) as exc_info:
            gen.throw(RuntimeError('trigger'))
        assert exc_info.value.args == ('converted from RuntimeError',)

    def test_generator_throw_with_target_type(self) -> None:
        """Throwing the target exception type into a generator -- double-wrap guard."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.echo_gen()
        next(gen)
        # Throwing AppError: generator doesn't catch it, it propagates.
        # The boundary sees it's already AppError and passes through.
        with pytest.raises(AppError) as exc_info:
            gen.throw(AppError('already target'))
        assert exc_info.value.args == ('already target',)
        assert exc_info.value.__cause__ is None

    def test_stopiteration_passthrough(self) -> None:
        """StopIteration from exhausted generator is not translated."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.good_items()
        assert next(gen) == 10
        assert next(gen) == 20
        assert next(gen) == 30
        with pytest.raises(StopIteration) as exc_info:
            next(gen)
        assert exc_info.value.args == ()

    def test_generator_body_failure_before_first_yield(self) -> None:
        """Exception before the first yield is translated at first next() call.

        Generator body doesn't execute until iteration starts. The
        _WrappedGenerator.__next__ calls next() inside the boundary,
        catching the ValueError from the generator body.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        gen = proxy.fail_during_creation()
        with pytest.raises(AppError) as exc_info:
            next(gen)
        assert exc_info.value.args == ('creation failure',)


class TestProxyAsyncGenerator:
    """Verify async generator wrapping translates exceptions during iteration."""

    async def test_async_generator_translates(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            async for _ in proxy.async_items():
                pass
        assert exc_info.value.args == ('async mid-iteration failure',)

    async def test_async_generator_success(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        result = [item async for item in proxy.async_good_items()]
        assert result == [10, 20]

    async def test_async_generator_stop_async_iteration_passthrough(self) -> None:
        """StopAsyncIteration from exhausted async generator is not translated."""
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        ait = proxy.async_good_items()
        assert await ait.__anext__() == 10
        assert await ait.__anext__() == 20
        with pytest.raises(StopAsyncIteration) as exc_info:
            await ait.__anext__()
        assert exc_info.value.args == ()


class TestProxyContextManager:
    """Verify context manager wrapping translates exceptions in __enter__/__exit__."""

    def test_cm_enter_translated(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info, proxy.connect_failing():
            pass
        assert exc_info.value.args == ('connect failed',)

    def test_cm_success(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with proxy.connect() as resource:
            assert resource == 'connected'

    def test_cm_body_exception_not_translated(self) -> None:
        """Exceptions from user code inside with-body are NOT translated.

        The body exception is passed to __exit__ as arguments, not raised
        inside the boundary. When __exit__ returns False, Python re-raises
        the original body exception outside the boundary.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(RuntimeError) as exc_info, proxy.connect():
            raise RuntimeError('user code')
        assert exc_info.value.args == ('user code',)

    def test_cm_exit_exception_translated(self) -> None:
        """Exception raised by __exit__ itself IS translated.

        The _WrappedContextManager calls __exit__ inside the boundary, so
        a library exception during cleanup is caught and translated.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info, proxy.connect_exit_failing():
            pass  # no body exception, but __exit__ raises
        assert exc_info.value.args == ('disconnect failed',)

    def test_cm_exit_raises_on_body_error(self) -> None:
        """__exit__ raises NEW exception while handling body exception.

        The body exception triggers __exit__, which raises its own error.
        The boundary translates the __exit__ error. The original body
        exception is lost (standard Python behavior: __exit__ exception
        replaces the body exception).
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info, proxy.connect_exit_raises_on_body_error():
            raise RuntimeError('body error')
        assert exc_info.value.args == ('cleanup failed during error handling',)

    def test_cm_exit_returning_true_suppresses(self) -> None:
        """__exit__ returning True suppresses the body exception.

        The _WrappedContextManager forwards the return value from __exit__,
        and Python suppresses the exception when it's True.
        """
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        # This should NOT raise -- __exit__ returns True, suppressing the exception
        with proxy.connect_suppressing():
            raise RuntimeError('suppressed by CM')

    async def test_async_cm_enter_translated(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        with pytest.raises(AppError) as exc_info:
            async with proxy.async_connect_failing():
                pass
        assert exc_info.value.args == ('async connect failed',)

    async def test_async_cm_success(self) -> None:
        proxy = LibraryBoundary(AppError).wrap(fake_lib)
        async with proxy.async_connect() as resource:
            assert resource == 'async connected'


class TestTargetExceptionVariants:
    """Verify behavior with unusual target exception types."""

    def test_target_with_custom_init(self) -> None:
        """Target with extra kwargs still works -- str(exc) becomes first arg."""
        boundary = LibraryBoundary(CustomInitError)
        with pytest.raises(CustomInitError) as exc_info, boundary:
            raise ValueError('oops')
        assert exc_info.value.args == ('oops',)
        assert exc_info.value.code == 0  # default value

    def test_target_no_arg_init(self) -> None:
        """Target whose __init__ takes no args -- str(exc) is passed, causes TypeError.

        This is a documented limitation: the target must accept a string message.
        The TypeError from __init__ propagates instead of the translated exception.
        """
        boundary = LibraryBoundary(NoArgError)
        with pytest.raises(TypeError) as exc_info, boundary:
            raise ValueError('oops')
        assert exc_info.value.args == ('NoArgError.__init__() takes 1 positional argument but 2 were given',)


class TestWithBashlex:
    """Verify LibraryBoundary translates real bashlex exceptions."""

    class BashlexError(Exception):
        """Test target for bashlex exceptions."""

    def test_parsing_error_translated(self) -> None:
        boundary = LibraryBoundary(self.BashlexError)
        with pytest.raises(self.BashlexError) as exc_info, boundary:
            bashlex.parse('((')
        assert isinstance(exc_info.value.__cause__, bashlex.errors.ParsingError)

    def test_not_implemented_translated(self) -> None:
        boundary = LibraryBoundary(self.BashlexError)
        with pytest.raises(self.BashlexError) as exc_info, boundary:
            bashlex.parse('echo $((1+2))')
        assert isinstance(exc_info.value.__cause__, NotImplementedError)

    def test_proxy_mode(self) -> None:
        """Fire-and-forget proxy wrapping translates bashlex exceptions."""
        bashlex_wrapped = LibraryBoundary(self.BashlexError).wrap(bashlex)
        with pytest.raises(self.BashlexError) as exc_info:
            bashlex_wrapped.parse('echo $((1+2))')
        assert exc_info.value.args == ('arithmetic expansion',)
        assert isinstance(exc_info.value.__cause__, NotImplementedError)
