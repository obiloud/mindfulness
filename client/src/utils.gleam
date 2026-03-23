import lustre/effect.{type Effect}

@external(javascript, "./ffi/timer_ffi.mjs", "setTimeout")
fn do_delay(ms: Int, func: fn() -> Nil) -> Nil

pub fn delay_effect(millis: Int, msg: msg) -> Effect(msg) {
  effect.from(fn(dispatch) { do_delay(millis, fn() { dispatch(msg) }) })
}
