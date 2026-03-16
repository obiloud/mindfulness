import gleam/regexp
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

pub fn sync_body_class(theme_class: String) -> effect.Effect(msg) {
  effect.from(fn(_) {
    let body = document.body()
    let css_class = element.get_attribute(body, "class")
    case css_class {
      Ok(class_name) -> {
        let assert Ok(pattern) = regexp.from_string("^(dark|light)\\s?")
        let new_class =
          regexp.replace(in: class_name, each: pattern, with: theme_class)
        element.set_attribute(body, "class", new_class)
      }

      Error(_) -> element.set_attribute(body, "class", theme_class)
    }
  })
}
