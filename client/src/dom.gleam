import lustre/effect
import plinth/browser/document
import plinth/browser/element
import plinth/browser/window

pub fn scroll_to_bottom(id: String) -> effect.Effect(msg) {
  effect.from(fn(_) {
    case document.get_element_by_id(id) {
      Ok(el) -> {
        let h = element.scroll_height(el)
        element.set_scroll_top(el, h)
      }
      Error(_) -> Nil
    }
  })
}

pub fn scroll_to_bottom_delayed(id: String) -> effect.Effect(msg) {
  effect.from(fn(_) {
    window.request_animation_frame(fn(_) {
      case document.get_element_by_id(id) {
        Ok(el) -> element.scroll_into_view(el)
        Error(_) -> Nil
      }
    })
    Nil
  })
}
