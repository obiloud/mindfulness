import gleam/dict.{type Dict}
import gleam/dynamic.{type Dynamic}
import gleam/dynamic/decode
import gleam/option.{type Option, None, Some}
import gleam/result

@external(javascript, "./ffi/utils.mjs", "is_null")
fn is_null(a: Dynamic) -> Bool

@external(javascript, "./ffi/local_storage.mjs", "get_item")
fn get_item_raw(key: String) -> Dynamic

pub fn get_item(key: String) -> Option(String) {
  let value = get_item_raw(key)
  case is_null(value) {
    True -> None
    False -> Some(decode.run(value, decode.string) |> result.unwrap(""))
  }
}

@external(javascript, "./ffi/local_storage.mjs", "set_item")
pub fn set_item(key: String, value: String) -> Bool

@external(javascript, "./ffi/local_storage.mjs", "remove_item")
pub fn remove_item(key: String) -> Bool

@external(javascript, "./ffi/local_storage.mjs", "clear_storage")
pub fn clear_storage() -> Bool

@external(javascript, "./ffi/local_storage.mjs", "has_item")
pub fn has_item(key: String) -> Bool

@external(javascript, "./ffi/local_storage.mjs", "get_all_keys")
pub fn get_all_keys() -> List(String)

@external(javascript, "./ffi/local_storage.mjs", "get_all_items")
pub fn get_all_items() -> Dict(String, Dynamic)

@external(javascript, "./ffi/local_storage.mjs", "get_storage_size")
pub fn get_storage_size() -> Int
