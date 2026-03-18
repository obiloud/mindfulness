export function auto_height(id) {
    let el = document.getElementById(id)
    const current_height = el.style.height;
    el.style.height = 'auto';
    if (el.scrollHeight < 200) {
        el.style.height = el.scrollHeight + 'px';
    } else {
        el.style.height = current_height;
    }
}

export function reset_height(id) {
    let el = document.getElementById(id)
    el.style.height = '';
}