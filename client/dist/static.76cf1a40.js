// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles

(function (
  modules,
  entry,
  mainEntry,
  parcelRequireName,
  externals,
  distDir,
  publicUrl,
  devServer
) {
  /* eslint-disable no-undef */
  var globalObject =
    typeof globalThis !== 'undefined'
      ? globalThis
      : typeof self !== 'undefined'
      ? self
      : typeof window !== 'undefined'
      ? window
      : typeof global !== 'undefined'
      ? global
      : {};
  /* eslint-enable no-undef */

  // Save the require from previous bundle to this closure if any
  var previousRequire =
    typeof globalObject[parcelRequireName] === 'function' &&
    globalObject[parcelRequireName];

  var importMap = previousRequire.i || {};
  var cache = previousRequire.cache || {};
  // Do not use `require` to prevent Webpack from trying to bundle this call
  var nodeRequire =
    typeof module !== 'undefined' &&
    typeof module.require === 'function' &&
    module.require.bind(module);

  function newRequire(name, jumped) {
    if (!cache[name]) {
      if (!modules[name]) {
        if (externals[name]) {
          return externals[name];
        }
        // if we cannot find the module within our internal map or
        // cache jump to the current global require ie. the last bundle
        // that was added to the page.
        var currentRequire =
          typeof globalObject[parcelRequireName] === 'function' &&
          globalObject[parcelRequireName];
        if (!jumped && currentRequire) {
          return currentRequire(name, true);
        }

        // If there are other bundles on this page the require from the
        // previous one is saved to 'previousRequire'. Repeat this as
        // many times as there are bundles until the module is found or
        // we exhaust the require chain.
        if (previousRequire) {
          return previousRequire(name, true);
        }

        // Try the node require function if it exists.
        if (nodeRequire && typeof name === 'string') {
          return nodeRequire(name);
        }

        var err = new Error("Cannot find module '" + name + "'");
        err.code = 'MODULE_NOT_FOUND';
        throw err;
      }

      localRequire.resolve = resolve;
      localRequire.cache = {};

      var module = (cache[name] = new newRequire.Module(name));

      modules[name][0].call(
        module.exports,
        localRequire,
        module,
        module.exports,
        globalObject
      );
    }

    return cache[name].exports;

    function localRequire(x) {
      var res = localRequire.resolve(x);
      if (res === false) {
        return {};
      }
      // Synthesize a module to follow re-exports.
      if (Array.isArray(res)) {
        var m = {__esModule: true};
        res.forEach(function (v) {
          var key = v[0];
          var id = v[1];
          var exp = v[2] || v[0];
          var x = newRequire(id);
          if (key === '*') {
            Object.keys(x).forEach(function (key) {
              if (
                key === 'default' ||
                key === '__esModule' ||
                Object.prototype.hasOwnProperty.call(m, key)
              ) {
                return;
              }

              Object.defineProperty(m, key, {
                enumerable: true,
                get: function () {
                  return x[key];
                },
              });
            });
          } else if (exp === '*') {
            Object.defineProperty(m, key, {
              enumerable: true,
              value: x,
            });
          } else {
            Object.defineProperty(m, key, {
              enumerable: true,
              get: function () {
                if (exp === 'default') {
                  return x.__esModule ? x.default : x;
                }
                return x[exp];
              },
            });
          }
        });
        return m;
      }
      return newRequire(res);
    }

    function resolve(x) {
      var id = modules[name][1][x];
      return id != null ? id : x;
    }
  }

  function Module(moduleName) {
    this.id = moduleName;
    this.bundle = newRequire;
    this.require = nodeRequire;
    this.exports = {};
  }

  newRequire.isParcelRequire = true;
  newRequire.Module = Module;
  newRequire.modules = modules;
  newRequire.cache = cache;
  newRequire.parent = previousRequire;
  newRequire.distDir = distDir;
  newRequire.publicUrl = publicUrl;
  newRequire.devServer = devServer;
  newRequire.i = importMap;
  newRequire.register = function (id, exports) {
    modules[id] = [
      function (require, module) {
        module.exports = exports;
      },
      {},
    ];
  };

  // Only insert newRequire.load when it is actually used.
  // The code in this file is linted against ES5, so dynamic import is not allowed.
  // INSERT_LOAD_HERE

  Object.defineProperty(newRequire, 'root', {
    get: function () {
      return globalObject[parcelRequireName];
    },
  });

  globalObject[parcelRequireName] = newRequire;

  for (var i = 0; i < entry.length; i++) {
    newRequire(entry[i]);
  }

  if (mainEntry) {
    // Expose entry point to Node, AMD or browser globals
    // Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
    var mainExports = newRequire(mainEntry);

    // CommonJS
    if (typeof exports === 'object' && typeof module !== 'undefined') {
      module.exports = mainExports;

      // RequireJS
    } else if (typeof define === 'function' && define.amd) {
      define(function () {
        return mainExports;
      });
    }
  }
})({"aZbtf":[function(require,module,exports,__globalThis) {
var global = arguments[3];
var HMR_HOST = null;
var HMR_PORT = null;
var HMR_SERVER_PORT = 8084;
var HMR_SECURE = false;
var HMR_ENV_HASH = "439701173a9199ea";
var HMR_USE_SSE = false;
module.bundle.HMR_BUNDLE_ID = "6f14b2c876cf1a40";
"use strict";
/* global HMR_HOST, HMR_PORT, HMR_SERVER_PORT, HMR_ENV_HASH, HMR_SECURE, HMR_USE_SSE, chrome, browser, __parcel__import__, __parcel__importScripts__, ServiceWorkerGlobalScope */ /*::
import type {
  HMRAsset,
  HMRMessage,
} from '@parcel/reporter-dev-server/src/HMRServer.js';
interface ParcelRequire {
  (string): mixed;
  cache: {|[string]: ParcelModule|};
  hotData: {|[string]: mixed|};
  Module: any;
  parent: ?ParcelRequire;
  isParcelRequire: true;
  modules: {|[string]: [Function, {|[string]: string|}]|};
  HMR_BUNDLE_ID: string;
  root: ParcelRequire;
}
interface ParcelModule {
  hot: {|
    data: mixed,
    accept(cb: (Function) => void): void,
    dispose(cb: (mixed) => void): void,
    // accept(deps: Array<string> | string, cb: (Function) => void): void,
    // decline(): void,
    _acceptCallbacks: Array<(Function) => void>,
    _disposeCallbacks: Array<(mixed) => void>,
  |};
}
interface ExtensionContext {
  runtime: {|
    reload(): void,
    getURL(url: string): string;
    getManifest(): {manifest_version: number, ...};
  |};
}
declare var module: {bundle: ParcelRequire, ...};
declare var HMR_HOST: string;
declare var HMR_PORT: string;
declare var HMR_SERVER_PORT: string;
declare var HMR_ENV_HASH: string;
declare var HMR_SECURE: boolean;
declare var HMR_USE_SSE: boolean;
declare var chrome: ExtensionContext;
declare var browser: ExtensionContext;
declare var __parcel__import__: (string) => Promise<void>;
declare var __parcel__importScripts__: (string) => Promise<void>;
declare var globalThis: typeof self;
declare var ServiceWorkerGlobalScope: Object;
*/ var OVERLAY_ID = '__parcel__error__overlay__';
var OldModule = module.bundle.Module;
function Module(moduleName) {
    OldModule.call(this, moduleName);
    this.hot = {
        data: module.bundle.hotData[moduleName],
        _acceptCallbacks: [],
        _disposeCallbacks: [],
        accept: function(fn) {
            this._acceptCallbacks.push(fn || function() {});
        },
        dispose: function(fn) {
            this._disposeCallbacks.push(fn);
        }
    };
    module.bundle.hotData[moduleName] = undefined;
}
module.bundle.Module = Module;
module.bundle.hotData = {};
var checkedAssets /*: {|[string]: boolean|} */ , disposedAssets /*: {|[string]: boolean|} */ , assetsToDispose /*: Array<[ParcelRequire, string]> */ , assetsToAccept /*: Array<[ParcelRequire, string]> */ , bundleNotFound = false;
function getHostname() {
    return HMR_HOST || (typeof location !== 'undefined' && location.protocol.indexOf('http') === 0 ? location.hostname : 'localhost');
}
function getPort() {
    return HMR_PORT || (typeof location !== 'undefined' ? location.port : HMR_SERVER_PORT);
}
// eslint-disable-next-line no-redeclare
let WebSocket = globalThis.WebSocket;
if (!WebSocket && typeof module.bundle.root === 'function') try {
    // eslint-disable-next-line no-global-assign
    WebSocket = module.bundle.root('ws');
} catch  {
// ignore.
}
var hostname = getHostname();
var port = getPort();
var protocol = HMR_SECURE || typeof location !== 'undefined' && location.protocol === 'https:' && ![
    'localhost',
    '127.0.0.1',
    '0.0.0.0'
].includes(hostname) ? 'wss' : 'ws';
// eslint-disable-next-line no-redeclare
var parent = module.bundle.parent;
if (!parent || !parent.isParcelRequire) {
    // Web extension context
    var extCtx = typeof browser === 'undefined' ? typeof chrome === 'undefined' ? null : chrome : browser;
    // Safari doesn't support sourceURL in error stacks.
    // eval may also be disabled via CSP, so do a quick check.
    var supportsSourceURL = false;
    try {
        (0, eval)('throw new Error("test"); //# sourceURL=test.js');
    } catch (err) {
        supportsSourceURL = err.stack.includes('test.js');
    }
    var ws;
    if (HMR_USE_SSE) ws = new EventSource('/__parcel_hmr');
    else try {
        // If we're running in the dev server's node runner, listen for messages on the parent port.
        let { workerData, parentPort } = module.bundle.root('node:worker_threads') /*: any*/ ;
        if (workerData !== null && workerData !== void 0 && workerData.__parcel) {
            parentPort.on('message', async (message)=>{
                try {
                    await handleMessage(message);
                    parentPort.postMessage('updated');
                } catch  {
                    parentPort.postMessage('restart');
                }
            });
            // After the bundle has finished running, notify the dev server that the HMR update is complete.
            queueMicrotask(()=>parentPort.postMessage('ready'));
        }
    } catch  {
        if (typeof WebSocket !== 'undefined') try {
            ws = new WebSocket(protocol + '://' + hostname + (port ? ':' + port : '') + '/');
        } catch (err) {
            // Ignore cloudflare workers error.
            if (err.message && !err.message.includes('Disallowed operation called within global scope')) console.error(err.message);
        }
    }
    if (ws) {
        // $FlowFixMe
        ws.onmessage = async function(event /*: {data: string, ...} */ ) {
            var data /*: HMRMessage */  = JSON.parse(event.data);
            await handleMessage(data);
        };
        if (ws instanceof WebSocket) {
            ws.onerror = function(e) {
                if (e.message) console.error(e.message);
            };
            ws.onclose = function() {
                console.warn("[parcel] \uD83D\uDEA8 Connection to the HMR server was lost");
            };
        }
    }
}
async function handleMessage(data /*: HMRMessage */ ) {
    checkedAssets = {} /*: {|[string]: boolean|} */ ;
    disposedAssets = {} /*: {|[string]: boolean|} */ ;
    assetsToAccept = [];
    assetsToDispose = [];
    bundleNotFound = false;
    if (data.type === 'reload') fullReload();
    else if (data.type === 'update') {
        // Remove error overlay if there is one
        if (typeof document !== 'undefined') removeErrorOverlay();
        let assets = data.assets;
        // Handle HMR Update
        let handled = assets.every((asset)=>{
            return asset.type === 'css' || asset.type === 'js' && hmrAcceptCheck(module.bundle.root, asset.id, asset.depsByBundle);
        });
        // Dispatch a custom event in case a bundle was not found. This might mean
        // an asset on the server changed and we should reload the page. This event
        // gives the client an opportunity to refresh without losing state
        // (e.g. via React Server Components). If e.preventDefault() is not called,
        // we will trigger a full page reload.
        if (handled && bundleNotFound && assets.some((a)=>a.envHash !== HMR_ENV_HASH) && typeof window !== 'undefined' && typeof CustomEvent !== 'undefined') handled = !window.dispatchEvent(new CustomEvent('parcelhmrreload', {
            cancelable: true
        }));
        if (handled) {
            console.clear();
            // Dispatch custom event so other runtimes (e.g React Refresh) are aware.
            if (typeof window !== 'undefined' && typeof CustomEvent !== 'undefined') window.dispatchEvent(new CustomEvent('parcelhmraccept'));
            await hmrApplyUpdates(assets);
            hmrDisposeQueue();
            // Run accept callbacks. This will also re-execute other disposed assets in topological order.
            let processedAssets = {};
            for(let i = 0; i < assetsToAccept.length; i++){
                let id = assetsToAccept[i][1];
                if (!processedAssets[id]) {
                    hmrAccept(assetsToAccept[i][0], id);
                    processedAssets[id] = true;
                }
            }
        } else fullReload();
    }
    if (data.type === 'error') {
        // Log parcel errors to console
        for (let ansiDiagnostic of data.diagnostics.ansi){
            let stack = ansiDiagnostic.codeframe ? ansiDiagnostic.codeframe : ansiDiagnostic.stack;
            console.error("\uD83D\uDEA8 [parcel]: " + ansiDiagnostic.message + '\n' + stack + '\n\n' + ansiDiagnostic.hints.join('\n'));
        }
        if (typeof document !== 'undefined') {
            // Render the fancy html overlay
            removeErrorOverlay();
            var overlay = createErrorOverlay(data.diagnostics.html);
            // $FlowFixMe
            document.body.appendChild(overlay);
        }
    }
}
function removeErrorOverlay() {
    var overlay = document.getElementById(OVERLAY_ID);
    if (overlay) {
        overlay.remove();
        console.log("[parcel] \u2728 Error resolved");
    }
}
function createErrorOverlay(diagnostics) {
    var overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    let errorHTML = '<div style="background: black; opacity: 0.85; font-size: 16px; color: white; position: fixed; height: 100%; width: 100%; top: 0px; left: 0px; padding: 30px; font-family: Menlo, Consolas, monospace; z-index: 9999;">';
    for (let diagnostic of diagnostics){
        let stack = diagnostic.frames.length ? diagnostic.frames.reduce((p, frame)=>{
            return `${p}
<a href="${protocol === 'wss' ? 'https' : 'http'}://${hostname}:${port}/__parcel_launch_editor?file=${encodeURIComponent(frame.location)}" style="text-decoration: underline; color: #888" onclick="fetch(this.href); return false">${frame.location}</a>
${frame.code}`;
        }, '') : diagnostic.stack;
        errorHTML += `
      <div>
        <div style="font-size: 18px; font-weight: bold; margin-top: 20px;">
          \u{1F6A8} ${diagnostic.message}
        </div>
        <pre>${stack}</pre>
        <div>
          ${diagnostic.hints.map((hint)=>"<div>\uD83D\uDCA1 " + hint + '</div>').join('')}
        </div>
        ${diagnostic.documentation ? `<div>\u{1F4DD} <a style="color: violet" href="${diagnostic.documentation}" target="_blank">Learn more</a></div>` : ''}
      </div>
    `;
    }
    errorHTML += '</div>';
    overlay.innerHTML = errorHTML;
    return overlay;
}
function fullReload() {
    if (typeof location !== 'undefined' && 'reload' in location) location.reload();
    else if (typeof extCtx !== 'undefined' && extCtx && extCtx.runtime && extCtx.runtime.reload) extCtx.runtime.reload();
    else try {
        let { workerData, parentPort } = module.bundle.root('node:worker_threads') /*: any*/ ;
        if (workerData !== null && workerData !== void 0 && workerData.__parcel) parentPort.postMessage('restart');
    } catch (err) {
        console.error("[parcel] \u26A0\uFE0F An HMR update was not accepted. Please restart the process.");
    }
}
function getParents(bundle, id) /*: Array<[ParcelRequire, string]> */ {
    var modules = bundle.modules;
    if (!modules) return [];
    var parents = [];
    var k, d, dep;
    for(k in modules)for(d in modules[k][1]){
        dep = modules[k][1][d];
        if (dep === id || Array.isArray(dep) && dep[dep.length - 1] === id) parents.push([
            bundle,
            k
        ]);
    }
    if (bundle.parent) parents = parents.concat(getParents(bundle.parent, id));
    return parents;
}
function updateLink(link) {
    var href = link.getAttribute('href');
    if (!href) return;
    var newLink = link.cloneNode();
    newLink.onload = function() {
        if (link.parentNode !== null) // $FlowFixMe
        link.parentNode.removeChild(link);
    };
    newLink.setAttribute('href', // $FlowFixMe
    href.split('?')[0] + '?' + Date.now());
    // $FlowFixMe
    link.parentNode.insertBefore(newLink, link.nextSibling);
}
var cssTimeout = null;
function reloadCSS() {
    if (cssTimeout || typeof document === 'undefined') return;
    cssTimeout = setTimeout(function() {
        var links = document.querySelectorAll('link[rel="stylesheet"]');
        for(var i = 0; i < links.length; i++){
            // $FlowFixMe[incompatible-type]
            var href /*: string */  = links[i].getAttribute('href');
            var hostname = getHostname();
            var servedFromHMRServer = hostname === 'localhost' ? new RegExp('^(https?:\\/\\/(0.0.0.0|127.0.0.1)|localhost):' + getPort()).test(href) : href.indexOf(hostname + ':' + getPort());
            var absolute = /^https?:\/\//i.test(href) && href.indexOf(location.origin) !== 0 && !servedFromHMRServer;
            if (!absolute) updateLink(links[i]);
        }
        cssTimeout = null;
    }, 50);
}
function hmrDownload(asset) {
    if (asset.type === 'js') {
        if (typeof document !== 'undefined') {
            let script = document.createElement('script');
            script.src = asset.url + '?t=' + Date.now();
            if (asset.outputFormat === 'esmodule') script.type = 'module';
            return new Promise((resolve, reject)=>{
                var _document$head;
                script.onload = ()=>resolve(script);
                script.onerror = reject;
                (_document$head = document.head) === null || _document$head === void 0 || _document$head.appendChild(script);
            });
        } else if (typeof importScripts === 'function') {
            // Worker scripts
            if (asset.outputFormat === 'esmodule') return import(asset.url + '?t=' + Date.now());
            else return new Promise((resolve, reject)=>{
                try {
                    importScripts(asset.url + '?t=' + Date.now());
                    resolve();
                } catch (err) {
                    reject(err);
                }
            });
        }
    }
}
async function hmrApplyUpdates(assets) {
    global.parcelHotUpdate = Object.create(null);
    let scriptsToRemove;
    try {
        // If sourceURL comments aren't supported in eval, we need to load
        // the update from the dev server over HTTP so that stack traces
        // are correct in errors/logs. This is much slower than eval, so
        // we only do it if needed (currently just Safari).
        // https://bugs.webkit.org/show_bug.cgi?id=137297
        // This path is also taken if a CSP disallows eval.
        if (!supportsSourceURL) {
            let promises = assets.map((asset)=>{
                var _hmrDownload;
                return (_hmrDownload = hmrDownload(asset)) === null || _hmrDownload === void 0 ? void 0 : _hmrDownload.catch((err)=>{
                    // Web extension fix
                    if (extCtx && extCtx.runtime && extCtx.runtime.getManifest().manifest_version == 3 && typeof ServiceWorkerGlobalScope != 'undefined' && global instanceof ServiceWorkerGlobalScope) {
                        extCtx.runtime.reload();
                        return;
                    }
                    throw err;
                });
            });
            scriptsToRemove = await Promise.all(promises);
        }
        assets.forEach(function(asset) {
            hmrApply(module.bundle.root, asset);
        });
    } finally{
        delete global.parcelHotUpdate;
        if (scriptsToRemove) scriptsToRemove.forEach((script)=>{
            if (script) {
                var _document$head2;
                (_document$head2 = document.head) === null || _document$head2 === void 0 || _document$head2.removeChild(script);
            }
        });
    }
}
function hmrApply(bundle /*: ParcelRequire */ , asset /*:  HMRAsset */ ) {
    var modules = bundle.modules;
    if (!modules) return;
    if (asset.type === 'css') reloadCSS();
    else if (asset.type === 'js') {
        let deps = asset.depsByBundle[bundle.HMR_BUNDLE_ID];
        if (deps) {
            if (modules[asset.id]) {
                // Remove dependencies that are removed and will become orphaned.
                // This is necessary so that if the asset is added back again, the cache is gone, and we prevent a full page reload.
                let oldDeps = modules[asset.id][1];
                for(let dep in oldDeps)if (!deps[dep] || deps[dep] !== oldDeps[dep]) {
                    let id = oldDeps[dep];
                    let parents = getParents(module.bundle.root, id);
                    if (parents.length === 1) hmrDelete(module.bundle.root, id);
                }
            }
            if (supportsSourceURL) // Global eval. We would use `new Function` here but browser
            // support for source maps is better with eval.
            (0, eval)(asset.output);
            // $FlowFixMe
            let fn = global.parcelHotUpdate[asset.id];
            modules[asset.id] = [
                fn,
                deps
            ];
        }
        // Always traverse to the parent bundle, even if we already replaced the asset in this bundle.
        // This is required in case modules are duplicated. We need to ensure all instances have the updated code.
        if (bundle.parent) hmrApply(bundle.parent, asset);
    }
}
function hmrDelete(bundle, id) {
    let modules = bundle.modules;
    if (!modules) return;
    if (modules[id]) {
        // Collect dependencies that will become orphaned when this module is deleted.
        let deps = modules[id][1];
        let orphans = [];
        for(let dep in deps){
            let parents = getParents(module.bundle.root, deps[dep]);
            if (parents.length === 1) orphans.push(deps[dep]);
        }
        // Delete the module. This must be done before deleting dependencies in case of circular dependencies.
        delete modules[id];
        delete bundle.cache[id];
        // Now delete the orphans.
        orphans.forEach((id)=>{
            hmrDelete(module.bundle.root, id);
        });
    } else if (bundle.parent) hmrDelete(bundle.parent, id);
}
function hmrAcceptCheck(bundle /*: ParcelRequire */ , id /*: string */ , depsByBundle /*: ?{ [string]: { [string]: string } }*/ ) {
    checkedAssets = {};
    if (hmrAcceptCheckOne(bundle, id, depsByBundle)) return true;
    // Traverse parents breadth first. All possible ancestries must accept the HMR update, or we'll reload.
    let parents = getParents(module.bundle.root, id);
    let accepted = false;
    while(parents.length > 0){
        let v = parents.shift();
        let a = hmrAcceptCheckOne(v[0], v[1], null);
        if (a) // If this parent accepts, stop traversing upward, but still consider siblings.
        accepted = true;
        else if (a !== null) {
            // Otherwise, queue the parents in the next level upward.
            let p = getParents(module.bundle.root, v[1]);
            if (p.length === 0) {
                // If there are no parents, then we've reached an entry without accepting. Reload.
                accepted = false;
                break;
            }
            parents.push(...p);
        }
    }
    return accepted;
}
function hmrAcceptCheckOne(bundle /*: ParcelRequire */ , id /*: string */ , depsByBundle /*: ?{ [string]: { [string]: string } }*/ ) {
    var modules = bundle.modules;
    if (!modules) return;
    if (depsByBundle && !depsByBundle[bundle.HMR_BUNDLE_ID]) {
        // If we reached the root bundle without finding where the asset should go,
        // there's nothing to do. Mark as "accepted" so we don't reload the page.
        if (!bundle.parent) {
            bundleNotFound = true;
            return true;
        }
        return hmrAcceptCheckOne(bundle.parent, id, depsByBundle);
    }
    if (checkedAssets[id]) return null;
    checkedAssets[id] = true;
    var cached = bundle.cache[id];
    if (!cached) return true;
    assetsToDispose.push([
        bundle,
        id
    ]);
    if (cached && cached.hot && cached.hot._acceptCallbacks.length) {
        assetsToAccept.push([
            bundle,
            id
        ]);
        return true;
    }
    return false;
}
function hmrDisposeQueue() {
    // Dispose all old assets.
    for(let i = 0; i < assetsToDispose.length; i++){
        let id = assetsToDispose[i][1];
        if (!disposedAssets[id]) {
            hmrDispose(assetsToDispose[i][0], id);
            disposedAssets[id] = true;
        }
    }
    assetsToDispose = [];
}
function hmrDispose(bundle /*: ParcelRequire */ , id /*: string */ ) {
    var cached = bundle.cache[id];
    bundle.hotData[id] = {};
    if (cached && cached.hot) cached.hot.data = bundle.hotData[id];
    if (cached && cached.hot && cached.hot._disposeCallbacks.length) cached.hot._disposeCallbacks.forEach(function(cb) {
        cb(bundle.hotData[id]);
    });
    delete bundle.cache[id];
}
function hmrAccept(bundle /*: ParcelRequire */ , id /*: string */ ) {
    // Execute the module.
    bundle(id);
    // Run the accept callbacks in the new version of the module.
    var cached = bundle.cache[id];
    if (cached && cached.hot && cached.hot._acceptCallbacks.length) {
        let assetsToAlsoAccept = [];
        cached.hot._acceptCallbacks.forEach(function(cb) {
            let additionalAssets = cb(function() {
                return getParents(module.bundle.root, id);
            });
            if (Array.isArray(additionalAssets) && additionalAssets.length) assetsToAlsoAccept.push(...additionalAssets);
        });
        if (assetsToAlsoAccept.length) {
            let handled = assetsToAlsoAccept.every(function(a) {
                return hmrAcceptCheck(a[0], a[1]);
            });
            if (!handled) return fullReload();
            hmrDisposeQueue();
        }
    }
}

},{}],"9GtLI":[function(require,module,exports,__globalThis) {
var _clientMjs = require("../build/dev/javascript/mindfulness_client/client.mjs");
(0, _clientMjs.main)();

},{"../build/dev/javascript/mindfulness_client/client.mjs":"dv2fd"}],"dv2fd":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Message", ()=>Message);
parcelHelpers.export(exports, "Message$Message", ()=>Message$Message);
parcelHelpers.export(exports, "Message$isMessage", ()=>Message$isMessage);
parcelHelpers.export(exports, "Message$Message$role", ()=>Message$Message$role);
parcelHelpers.export(exports, "Message$Message$0", ()=>Message$Message$0);
parcelHelpers.export(exports, "Message$Message$content", ()=>Message$Message$content);
parcelHelpers.export(exports, "Message$Message$1", ()=>Message$Message$1);
parcelHelpers.export(exports, "AgentResponse", ()=>AgentResponse);
parcelHelpers.export(exports, "AgentResponse$AgentResponse", ()=>AgentResponse$AgentResponse);
parcelHelpers.export(exports, "AgentResponse$isAgentResponse", ()=>AgentResponse$isAgentResponse);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$session_id", ()=>AgentResponse$AgentResponse$session_id);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$0", ()=>AgentResponse$AgentResponse$0);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$message", ()=>AgentResponse$AgentResponse$message);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$1", ()=>AgentResponse$AgentResponse$1);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$transcript", ()=>AgentResponse$AgentResponse$transcript);
parcelHelpers.export(exports, "AgentResponse$AgentResponse$2", ()=>AgentResponse$AgentResponse$2);
parcelHelpers.export(exports, "Model", ()=>Model);
parcelHelpers.export(exports, "Model$Model", ()=>Model$Model);
parcelHelpers.export(exports, "Model$isModel", ()=>Model$isModel);
parcelHelpers.export(exports, "Model$Model$chat_history", ()=>Model$Model$chat_history);
parcelHelpers.export(exports, "Model$Model$0", ()=>Model$Model$0);
parcelHelpers.export(exports, "Model$Model$is_streaming", ()=>Model$Model$is_streaming);
parcelHelpers.export(exports, "Model$Model$1", ()=>Model$Model$1);
parcelHelpers.export(exports, "Model$Model$input_text", ()=>Model$Model$input_text);
parcelHelpers.export(exports, "Model$Model$2", ()=>Model$Model$2);
parcelHelpers.export(exports, "Model$Model$loading", ()=>Model$Model$loading);
parcelHelpers.export(exports, "Model$Model$3", ()=>Model$Model$3);
parcelHelpers.export(exports, "Model$Model$transcript", ()=>Model$Model$transcript);
parcelHelpers.export(exports, "Model$Model$4", ()=>Model$Model$4);
parcelHelpers.export(exports, "Model$Model$session_id", ()=>Model$Model$session_id);
parcelHelpers.export(exports, "Model$Model$5", ()=>Model$Model$5);
parcelHelpers.export(exports, "UserTyped", ()=>UserTyped);
parcelHelpers.export(exports, "Msg$UserTyped", ()=>Msg$UserTyped);
parcelHelpers.export(exports, "Msg$isUserTyped", ()=>Msg$isUserTyped);
parcelHelpers.export(exports, "Msg$UserTyped$0", ()=>Msg$UserTyped$0);
parcelHelpers.export(exports, "UserRequestedAudio", ()=>UserRequestedAudio);
parcelHelpers.export(exports, "Msg$UserRequestedAudio", ()=>Msg$UserRequestedAudio);
parcelHelpers.export(exports, "Msg$isUserRequestedAudio", ()=>Msg$isUserRequestedAudio);
parcelHelpers.export(exports, "AudioStarted", ()=>AudioStarted);
parcelHelpers.export(exports, "Msg$AudioStarted", ()=>Msg$AudioStarted);
parcelHelpers.export(exports, "Msg$isAudioStarted", ()=>Msg$isAudioStarted);
parcelHelpers.export(exports, "AudioEnded", ()=>AudioEnded);
parcelHelpers.export(exports, "Msg$AudioEnded", ()=>Msg$AudioEnded);
parcelHelpers.export(exports, "Msg$isAudioEnded", ()=>Msg$isAudioEnded);
parcelHelpers.export(exports, "ReceiveChatResponse", ()=>ReceiveChatResponse);
parcelHelpers.export(exports, "Msg$ReceiveChatResponse", ()=>Msg$ReceiveChatResponse);
parcelHelpers.export(exports, "Msg$isReceiveChatResponse", ()=>Msg$isReceiveChatResponse);
parcelHelpers.export(exports, "Msg$ReceiveChatResponse$0", ()=>Msg$ReceiveChatResponse$0);
parcelHelpers.export(exports, "SendMessage", ()=>SendMessage);
parcelHelpers.export(exports, "Msg$SendMessage", ()=>Msg$SendMessage);
parcelHelpers.export(exports, "Msg$isSendMessage", ()=>Msg$isSendMessage);
parcelHelpers.export(exports, "main", ()=>main);
var _listMjs = require("../gleam_stdlib/gleam/list.mjs");
var _optionMjs = require("../gleam_stdlib/gleam/option.mjs");
var _lustreMjs = require("../lustre/lustre.mjs");
var _attributeMjs = require("../lustre/lustre/attribute.mjs");
var _effectMjs = require("../lustre/lustre/effect.mjs");
var _elementMjs = require("../lustre/lustre/element.mjs");
var _htmlMjs = require("../lustre/lustre/element/html.mjs");
var _eventMjs = require("../lustre/lustre/event.mjs");
var _audioFfiMjs = require("./audio_ffi.mjs");
var _gleamMjs = require("./gleam.mjs");
const FILEPATH = "src/client.gleam";
class Message extends (0, _gleamMjs.CustomType) {
    constructor(role, content){
        super();
        this.role = role;
        this.content = content;
    }
}
const Message$Message = (role, content)=>new Message(role, content);
const Message$isMessage = (value)=>value instanceof Message;
const Message$Message$role = (value)=>value.role;
const Message$Message$0 = (value)=>value.role;
const Message$Message$content = (value)=>value.content;
const Message$Message$1 = (value)=>value.content;
class AgentResponse extends (0, _gleamMjs.CustomType) {
    constructor(session_id, message, transcript){
        super();
        this.session_id = session_id;
        this.message = message;
        this.transcript = transcript;
    }
}
const AgentResponse$AgentResponse = (session_id, message, transcript)=>new AgentResponse(session_id, message, transcript);
const AgentResponse$isAgentResponse = (value)=>value instanceof AgentResponse;
const AgentResponse$AgentResponse$session_id = (value)=>value.session_id;
const AgentResponse$AgentResponse$0 = (value)=>value.session_id;
const AgentResponse$AgentResponse$message = (value)=>value.message;
const AgentResponse$AgentResponse$1 = (value)=>value.message;
const AgentResponse$AgentResponse$transcript = (value)=>value.transcript;
const AgentResponse$AgentResponse$2 = (value)=>value.transcript;
class Model extends (0, _gleamMjs.CustomType) {
    constructor(chat_history, is_streaming, input_text, loading, transcript, session_id){
        super();
        this.chat_history = chat_history;
        this.is_streaming = is_streaming;
        this.input_text = input_text;
        this.loading = loading;
        this.transcript = transcript;
        this.session_id = session_id;
    }
}
const Model$Model = (chat_history, is_streaming, input_text, loading, transcript, session_id)=>new Model(chat_history, is_streaming, input_text, loading, transcript, session_id);
const Model$isModel = (value)=>value instanceof Model;
const Model$Model$chat_history = (value)=>value.chat_history;
const Model$Model$0 = (value)=>value.chat_history;
const Model$Model$is_streaming = (value)=>value.is_streaming;
const Model$Model$1 = (value)=>value.is_streaming;
const Model$Model$input_text = (value)=>value.input_text;
const Model$Model$2 = (value)=>value.input_text;
const Model$Model$loading = (value)=>value.loading;
const Model$Model$3 = (value)=>value.loading;
const Model$Model$transcript = (value)=>value.transcript;
const Model$Model$4 = (value)=>value.transcript;
const Model$Model$session_id = (value)=>value.session_id;
const Model$Model$5 = (value)=>value.session_id;
class UserTyped extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const Msg$UserTyped = ($0)=>new UserTyped($0);
const Msg$isUserTyped = (value)=>value instanceof UserTyped;
const Msg$UserTyped$0 = (value)=>value[0];
class UserRequestedAudio extends (0, _gleamMjs.CustomType) {
}
const Msg$UserRequestedAudio = ()=>new UserRequestedAudio();
const Msg$isUserRequestedAudio = (value)=>value instanceof UserRequestedAudio;
class AudioStarted extends (0, _gleamMjs.CustomType) {
}
const Msg$AudioStarted = ()=>new AudioStarted();
const Msg$isAudioStarted = (value)=>value instanceof AudioStarted;
class AudioEnded extends (0, _gleamMjs.CustomType) {
}
const Msg$AudioEnded = ()=>new AudioEnded();
const Msg$isAudioEnded = (value)=>value instanceof AudioEnded;
class ReceiveChatResponse extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const Msg$ReceiveChatResponse = ($0)=>new ReceiveChatResponse($0);
const Msg$isReceiveChatResponse = (value)=>value instanceof ReceiveChatResponse;
const Msg$ReceiveChatResponse$0 = (value)=>value[0];
class SendMessage extends (0, _gleamMjs.CustomType) {
}
const Msg$SendMessage = ()=>new SendMessage();
const Msg$isSendMessage = (value)=>value instanceof SendMessage;
function init(_) {
    return [
        new Model((0, _gleamMjs.toList)([]), false, "", false, new (0, _optionMjs.None)(), new (0, _optionMjs.None)()),
        _effectMjs.none()
    ];
}
function update(model, msg) {
    if (msg instanceof UserTyped) {
        let val = msg[0];
        return [
            new Model(model.chat_history, model.is_streaming, val, model.loading, model.transcript, model.session_id),
            _effectMjs.none()
        ];
    } else if (msg instanceof UserRequestedAudio) return [
        new Model(model.chat_history, true, model.input_text, model.loading, model.transcript, model.session_id),
        _effectMjs.from((_)=>{
            return (0, _audioFfiMjs.init_cartesia_stream)();
        })
    ];
    else if (msg instanceof AudioStarted) return [
        new Model(model.chat_history, true, model.input_text, model.loading, model.transcript, model.session_id),
        _effectMjs.none()
    ];
    else if (msg instanceof AudioEnded) return [
        new Model(model.chat_history, false, model.input_text, model.loading, model.transcript, model.session_id),
        _effectMjs.none()
    ];
    else if (msg instanceof ReceiveChatResponse) {
        let msg$1 = msg[0];
        return [
            new Model((0, _gleamMjs.prepend)(new Message("assistant", msg$1.message), model.chat_history), model.is_streaming, model.input_text, false, msg$1.transcript, new (0, _optionMjs.Some)(msg$1.session_id)),
            _effectMjs.none()
        ];
    } else return [
        new Model((0, _gleamMjs.prepend)(new Message("user", model.input_text), model.chat_history), false, "", true, model.transcript, model.session_id),
        _effectMjs.from((_)=>{
            return (0, _audioFfiMjs.init_cartesia_stream)();
        })
    ];
}
function view(model) {
    return _htmlMjs.div((0, _gleamMjs.toList)([
        _attributeMjs.class$("min-h-screen flex items-center justify-center px-4")
    ]), (0, _gleamMjs.toList)([
        _htmlMjs.div((0, _gleamMjs.toList)([
            _attributeMjs.class$("w-full max-w-3xl bg-mind-surface/80 backdrop-blur-md rounded-2xl shadow-xl border border-slate-700/60 p-6 flex flex-col gap-4")
        ]), (0, _gleamMjs.toList)([
            _htmlMjs.header((0, _gleamMjs.toList)([
                _attributeMjs.class$("flex items-center justify-between mb-2")
            ]), (0, _gleamMjs.toList)([
                _htmlMjs.div((0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                    _htmlMjs.h1((0, _gleamMjs.toList)([
                        _attributeMjs.class$("text-xl font-semibold text-sky-300")
                    ]), (0, _gleamMjs.toList)([
                        _elementMjs.text("Mindfulness AI")
                    ])),
                    _htmlMjs.p((0, _gleamMjs.toList)([
                        _attributeMjs.class$("text-sm text-slate-300")
                    ]), (0, _gleamMjs.toList)([
                        _elementMjs.text("Share how you feel, and receive a gentle, guided response.")
                    ]))
                ]))
            ])),
            _htmlMjs.div((0, _gleamMjs.toList)([
                _attributeMjs.class$("flex-1 min-h-[320px] max-h-[420px] overflow-y-auto space-y-3 pr-1")
            ]), (0, _gleamMjs.toList)([
                (()=>{
                    let $ = model.chat_history;
                    if ($ instanceof (0, _gleamMjs.Empty)) return _htmlMjs.p((0, _gleamMjs.toList)([
                        _attributeMjs.class$("text-slate-400 text-sm")
                    ]), (0, _gleamMjs.toList)([
                        _elementMjs.text("Start by telling the guide what you are going through, for example: \u201CI feel anxious about work and cannot relax.\u201D")
                    ]));
                    else return _htmlMjs.text("");
                })(),
                _htmlMjs.div((0, _gleamMjs.toList)([]), (()=>{
                    let _pipe = model.chat_history;
                    return _listMjs.map(_pipe, (m)=>{
                        let _block;
                        let $ = m.role;
                        if ($ === "user") _block = "justify-end";
                        else _block = "justify-start";
                        let justify_class = _block;
                        let _block$1;
                        let $1 = m.role;
                        if ($1 === "user") _block$1 = "bg-mind-accent/90 text-slate-950";
                        else _block$1 = "bg-slate-800/80 text-slate-100 border border-slate-700/70";
                        let bg_class = _block$1;
                        return _htmlMjs.div((0, _gleamMjs.toList)([
                            _attributeMjs.class$("flex " + justify_class)
                        ]), (0, _gleamMjs.toList)([
                            _htmlMjs.div((0, _gleamMjs.toList)([
                                _attributeMjs.class$("max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap " + bg_class)
                            ]), (0, _gleamMjs.toList)([
                                _elementMjs.text(m.content)
                            ]))
                        ]));
                    });
                })()),
                (()=>{
                    let $ = model.loading;
                    if ($) return _htmlMjs.div((0, _gleamMjs.toList)([
                        _attributeMjs.class$("flex justify-center items-center py-2")
                    ]), (0, _gleamMjs.toList)([
                        _htmlMjs.div((0, _gleamMjs.toList)([
                            _attributeMjs.class$("flex items-center space-x-2")
                        ]), (0, _gleamMjs.toList)([
                            _htmlMjs.div((0, _gleamMjs.toList)([
                                _attributeMjs.class$("w-2 h-2 bg-sky-400 rounded-full animate-pulse")
                            ]), (0, _gleamMjs.toList)([])),
                            _htmlMjs.div((0, _gleamMjs.toList)([
                                _attributeMjs.class$("w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-100")
                            ]), (0, _gleamMjs.toList)([])),
                            _htmlMjs.div((0, _gleamMjs.toList)([
                                _attributeMjs.class$("w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-200")
                            ]), (0, _gleamMjs.toList)([]))
                        ]))
                    ]));
                    else return _htmlMjs.text("");
                })()
            ])),
            _htmlMjs.div((0, _gleamMjs.toList)([
                _attributeMjs.class$("mt-2 flex gap-2")
            ]), (0, _gleamMjs.toList)([
                _htmlMjs.input((0, _gleamMjs.toList)([
                    _attributeMjs.type_("text"),
                    _eventMjs.on_change((var0)=>{
                        return new UserTyped(var0);
                    }),
                    _attributeMjs.placeholder("How are you feeling?"),
                    _attributeMjs.class$("flex-1 rounded-xl bg-slate-900/60 border border-slate-700/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-mind-accent focus:border-transparent")
                ])),
                _htmlMjs.button((0, _gleamMjs.toList)([
                    _eventMjs.on_click(new SendMessage()),
                    _attributeMjs.disabled(model.loading),
                    _attributeMjs.class$("px-4 py-2 rounded-xl bg-mind-accent text-slate-950 text-sm font-medium hover:bg-sky-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors")
                ]), (0, _gleamMjs.toList)([
                    _elementMjs.text((()=>{
                        let $ = model.loading;
                        if ($) return "Sending...";
                        else return "Send";
                    })())
                ]))
            ]))
        ]))
    ]));
}
function main() {
    let app = _lustreMjs.application(init, update, view);
    let $ = _lustreMjs.start(app, "#app", undefined);
    if (!($ instanceof (0, _gleamMjs.Ok))) throw (0, _gleamMjs.makeError)("let_assert", FILEPATH, "client", 276, "main", "Pattern match failed, no pattern matched the value.", {
        value: $,
        start: 8454,
        end: 8503,
        pattern_start: 8465,
        pattern_end: 8470
    });
    return $;
}

},{"../gleam_stdlib/gleam/list.mjs":"8dUwY","../lustre/lustre.mjs":"9FST8","../lustre/lustre/attribute.mjs":"faRXj","../lustre/lustre/effect.mjs":"iAEPi","../lustre/lustre/element.mjs":"2XxJ4","../lustre/lustre/element/html.mjs":"eLT3l","../lustre/lustre/event.mjs":"29g6I","./audio_ffi.mjs":"eHhmy","./gleam.mjs":"aBxRS","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT","../gleam_stdlib/gleam/option.mjs":"aWtoH"}],"8dUwY":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Continue", ()=>Continue);
parcelHelpers.export(exports, "ContinueOrStop$Continue", ()=>ContinueOrStop$Continue);
parcelHelpers.export(exports, "ContinueOrStop$isContinue", ()=>ContinueOrStop$isContinue);
parcelHelpers.export(exports, "ContinueOrStop$Continue$0", ()=>ContinueOrStop$Continue$0);
parcelHelpers.export(exports, "Stop", ()=>Stop);
parcelHelpers.export(exports, "ContinueOrStop$Stop", ()=>ContinueOrStop$Stop);
parcelHelpers.export(exports, "ContinueOrStop$isStop", ()=>ContinueOrStop$isStop);
parcelHelpers.export(exports, "ContinueOrStop$Stop$0", ()=>ContinueOrStop$Stop$0);
/**
 * Counts the number of elements in a given list.
 *
 * This function has to traverse the list to determine the number of elements,
 * so it runs in linear time.
 *
 * This function is natively implemented by the virtual machine and is highly
 * optimised.
 *
 * ## Examples
 *
 * ```gleam
 * assert length([]) == 0
 * ```
 *
 * ```gleam
 * assert length([1]) == 1
 * ```
 *
 * ```gleam
 * assert length([1, 2]) == 2
 * ```
 */ parcelHelpers.export(exports, "length", ()=>length);
/**
 * Counts the number of elements in a given list satisfying a given predicate.
 *
 * This function has to traverse the list to determine the number of elements,
 * so it runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert count([], fn(a) { a > 0 }) == 0
 * ```
 *
 * ```gleam
 * assert count([1], fn(a) { a > 0 }) == 1
 * ```
 *
 * ```gleam
 * assert count([1, 2, 3], int.is_odd) == 2
 * ```
 */ parcelHelpers.export(exports, "count", ()=>count);
/**
 * Creates a new list from a given list containing the same elements but in the
 * opposite order.
 *
 * This function has to traverse the list to create the new reversed list, so
 * it runs in linear time.
 *
 * This function is natively implemented by the virtual machine and is highly
 * optimised.
 *
 * ## Examples
 *
 * ```gleam
 * assert reverse([]) == []
 * ```
 *
 * ```gleam
 * assert reverse([1]) == [1]
 * ```
 *
 * ```gleam
 * assert reverse([1, 2]) == [2, 1]
 * ```
 */ parcelHelpers.export(exports, "reverse", ()=>reverse);
/**
 * Determines whether or not the list is empty.
 *
 * This function runs in constant time.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_empty([])
 * ```
 *
 * ```gleam
 * assert !is_empty([1])
 * ```
 *
 * ```gleam
 * assert !is_empty([1, 1])
 * ```
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
/**
 * Determines whether or not a given element exists within a given list.
 *
 * This function traverses the list to find the element, so it runs in linear
 * time.
 *
 * ## Examples
 *
 * ```gleam
 * assert !contains([], any: 0)
 * ```
 *
 * ```gleam
 * assert [0] |> contains(any: 0)
 * ```
 *
 * ```gleam
 * assert !contains([1], any: 0)
 * ```
 *
 * ```gleam
 * assert !contains([1, 1], any: 0)
 * ```
 *
 * ```gleam
 * assert [1, 0] |> contains(any: 0)
 * ```
 */ parcelHelpers.export(exports, "contains", ()=>contains);
/**
 * Gets the first element from the start of the list, if there is one.
 *
 * ## Examples
 *
 * ```gleam
 * assert first([]) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert first([0]) == Ok(0)
 * ```
 *
 * ```gleam
 * assert first([1, 2]) == Ok(1)
 * ```
 */ parcelHelpers.export(exports, "first", ()=>first);
/**
 * Returns the list minus the first element. If the list is empty, `Error(Nil)` is
 * returned.
 *
 * This function runs in constant time and does not make a copy of the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert rest([]) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert rest([0]) == Ok([])
 * ```
 *
 * ```gleam
 * assert rest([1, 2]) == Ok([2])
 * ```
 */ parcelHelpers.export(exports, "rest", ()=>rest);
/**
 * Groups the elements from the given list by the given key function.
 *
 * Does not preserve the initial value order.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/dict
 *
 * assert
 *   [Ok(3), Error("Wrong"), Ok(200), Ok(73)]
 *   |> group(by: fn(i) {
 *     case i {
 *       Ok(_) -> "Successful"
 *       Error(_) -> "Failed"
 *     }
 *   })
 *   |> dict.to_list
 *   == [
 *     #("Failed", [Error("Wrong")]),
 *     #("Successful", [Ok(73), Ok(200), Ok(3)])
 *   ]
 * ```
 *
 * ```gleam
 * import gleam/dict
 *
 * assert group([1,2,3,4,5], by: fn(i) { i - i / 3 * 3 })
 *   |> dict.to_list
 *   == [#(0, [3]), #(1, [4, 1]), #(2, [5, 2])]
 * ```
 */ parcelHelpers.export(exports, "group", ()=>group);
/**
 * Returns a new list containing only the elements from the first list for
 * which the given functions returns `True`.
 *
 * ## Examples
 *
 * ```gleam
 * assert filter([2, 4, 6, 1], fn(x) { x > 2 }) == [4, 6]
 * ```
 *
 * ```gleam
 * assert filter([2, 4, 6, 1], fn(x) { x > 6 }) == []
 * ```
 */ parcelHelpers.export(exports, "filter", ()=>filter);
/**
 * Returns a new list containing only the elements from the first list for
 * which the given functions returns `Ok(_)`.
 *
 * ## Examples
 *
 * ```gleam
 * assert filter_map([2, 4, 6, 1], Error) == []
 * ```
 *
 * ```gleam
 * assert filter_map([2, 4, 6, 1], fn(x) { Ok(x + 1) }) == [3, 5, 7, 2]
 * ```
 */ parcelHelpers.export(exports, "filter_map", ()=>filter_map);
/**
 * Returns a new list containing the results of applying the supplied function to each element.
 *
 * ## Examples
 *
 * ```gleam
 * assert map([2, 4, 6], fn(x) { x * 2 }) == [4, 8, 12]
 * ```
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Combines two lists into a single list using the given function.
 *
 * If a list is longer than the other, the extra elements are dropped.
 *
 * ## Examples
 *
 * ```gleam
 * assert map2([1, 2, 3], [4, 5, 6], fn(x, y) { x + y }) == [5, 7, 9]
 * ```
 *
 * ```gleam
 * assert map2([1, 2], ["a", "b", "c"], fn(i, x) { #(i, x) })
 *   == [#(1, "a"), #(2, "b")]
 * ```
 */ parcelHelpers.export(exports, "map2", ()=>map2);
/**
 * Similar to `map` but also lets you pass around an accumulated value.
 *
 * ## Examples
 *
 * ```gleam
 * assert
 *   map_fold(
 *     over: [1, 2, 3],
 *     from: 100,
 *     with: fn(memo, i) { #(memo + i, i * 2) }
 *   )
 *   == #(106, [2, 4, 6])
 * ```
 */ parcelHelpers.export(exports, "map_fold", ()=>map_fold);
/**
 * Similar to `map`, but the supplied function will also be passed the index
 * of the element being mapped as an additional argument.
 *
 * The index starts at 0, so the first element is 0, the second is 1, and so
 * on.
 *
 * ## Examples
 *
 * ```gleam
 * assert index_map(["a", "b"], fn(x, i) { #(i, x) }) == [#(0, "a"), #(1, "b")]
 * ```
 */ parcelHelpers.export(exports, "index_map", ()=>index_map);
/**
 * Takes a function that returns a `Result` and applies it to each element in a
 * given list in turn.
 *
 * If the function returns `Ok(new_value)` for all elements in the list then a
 * list of the new values is returned.
 *
 * If the function returns `Error(reason)` for any of the elements then it is
 * returned immediately. None of the elements in the list are processed after
 * one returns an `Error`.
 *
 * ## Examples
 *
 * ```gleam
 * assert try_map([1, 2, 3], fn(x) { Ok(x + 2) }) == Ok([3, 4, 5])
 * ```
 *
 * ```gleam
 * assert try_map([1, 2, 3], fn(_) { Error(0) }) == Error(0)
 * ```
 *
 * ```gleam
 * assert try_map([[1], [2, 3]], first) == Ok([1, 2])
 * ```
 *
 * ```gleam
 * assert try_map([[1], [], [2]], first) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "try_map", ()=>try_map);
/**
 * Returns a list that is the given list with up to the given number of
 * elements removed from the front of the list.
 *
 * If the list has less than the number of elements an empty list is
 * returned.
 *
 * This function runs in linear time but does not copy the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert drop([1, 2, 3, 4], 2) == [3, 4]
 * ```
 *
 * ```gleam
 * assert drop([1, 2, 3, 4], 9) == []
 * ```
 */ parcelHelpers.export(exports, "drop", ()=>drop);
/**
 * Returns a list containing the first given number of elements from the given
 * list.
 *
 * If the list has less than the number of elements then the full list is
 * returned.
 *
 * This function runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert take([1, 2, 3, 4], 2) == [1, 2]
 * ```
 *
 * ```gleam
 * assert take([1, 2, 3, 4], 9) == [1, 2, 3, 4]
 * ```
 */ parcelHelpers.export(exports, "take", ()=>take);
/**
 * Returns a new empty list.
 *
 * ## Examples
 *
 * ```gleam
 * assert new() == []
 * ```
 */ parcelHelpers.export(exports, "new$", ()=>new$);
/**
 * Returns the given item wrapped in a list.
 *
 * ## Examples
 *
 * ```gleam
 * assert wrap(1) == [1]
 * ```
 *
 * ```gleam
 * assert wrap(["a", "b", "c"]) == [["a", "b", "c"]]
 * ```
 *
 * ```gleam
 * assert wrap([[]]) == [[[]]]
 * ```
 */ parcelHelpers.export(exports, "wrap", ()=>wrap);
/**
 * Joins one list onto the end of another.
 *
 * This function runs in linear time, and it traverses and copies the first
 * list.
 *
 * ## Examples
 *
 * ```gleam
 * assert append([1, 2], [3]) == [1, 2, 3]
 * ```
 */ parcelHelpers.export(exports, "append", ()=>append);
/**
 * Prefixes an item to a list. This can also be done using the dedicated
 * syntax instead.
 *
 * ```gleam
 * let existing_list = [2, 3, 4]
 * assert [1, ..existing_list] == [1, 2, 3, 4]
 * ```
 *
 * ```gleam
 * let existing_list = [2, 3, 4]
 * assert prepend(to: existing_list, this: 1) == [1, 2, 3, 4]
 * ```
 */ parcelHelpers.export(exports, "prepend", ()=>prepend);
/**
 * Joins a list of lists into a single list.
 *
 * This function traverses all elements twice on the JavaScript target.
 * This function traverses all elements once on the Erlang target.
 *
 * ## Examples
 *
 * ```gleam
 * assert flatten([[1], [2, 3], []]) == [1, 2, 3]
 * ```
 */ parcelHelpers.export(exports, "flatten", ()=>flatten);
/**
 * Maps the list with the given function into a list of lists, and then flattens it.
 *
 * ## Examples
 *
 * ```gleam
 * assert flat_map([2, 4, 6], fn(x) { [x, x + 1] }) == [2, 3, 4, 5, 6, 7]
 * ```
 */ parcelHelpers.export(exports, "flat_map", ()=>flat_map);
/**
 * Reduces a list of elements into a single value by calling a given function
 * on each element, going from left to right.
 *
 * `fold([1, 2, 3], 0, add)` is the equivalent of
 * `add(add(add(0, 1), 2), 3)`.
 *
 * This function runs in linear time.
 */ parcelHelpers.export(exports, "fold", ()=>fold);
/**
 * Reduces a list of elements into a single value by calling a given function
 * on each element, going from right to left.
 *
 * `fold_right([1, 2, 3], 0, add)` is the equivalent of
 * `add(add(add(0, 3), 2), 1)`.
 *
 * This function runs in linear time.
 *
 * Unlike `fold` this function is not tail recursive. Where possible use
 * `fold` instead as it will use less memory.
 */ parcelHelpers.export(exports, "fold_right", ()=>fold_right);
/**
 * Like `fold` but the folding function also receives the index of the current element.
 *
 * ## Examples
 *
 * ```gleam
 * assert ["a", "b", "c"]
 *   |> index_fold("", fn(acc, item, index) {
 *     acc <> int.to_string(index) <> ":" <> item <> " "
 *   })
 *   == "0:a 1:b 2:c"
 * ```
 *
 * ```gleam
 * assert [10, 20, 30]
 *   |> index_fold(0, fn(acc, item, index) { acc + item * index })
 *   == 80
 * ```
 */ parcelHelpers.export(exports, "index_fold", ()=>index_fold);
/**
 * A variant of fold that might fail.
 *
 * The folding function should return `Result(accumulator, error)`.
 * If the returned value is `Ok(accumulator)` try_fold will try the next value in the list.
 * If the returned value is `Error(error)` try_fold will stop and return that error.
 *
 * ## Examples
 *
 * ```gleam
 * assert [1, 2, 3, 4]
 *   |> try_fold(0, fn(acc, i) {
 *     case i < 3 {
 *       True -> Ok(acc + i)
 *       False -> Error(Nil)
 *     }
 *   })
 *   == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "try_fold", ()=>try_fold);
/**
 * A variant of fold that allows to stop folding earlier.
 *
 * The folding function should return `ContinueOrStop(accumulator)`.
 * If the returned value is `Continue(accumulator)` fold_until will try the next value in the list.
 * If the returned value is `Stop(accumulator)` fold_until will stop and return that accumulator.
 *
 * ## Examples
 *
 * ```gleam
 * assert [1, 2, 3, 4]
 *   |> fold_until(0, fn(acc, i) {
 *     case i < 3 {
 *       True -> Continue(acc + i)
 *       False -> Stop(acc)
 *     }
 *   })
 *   == 3
 * ```
 */ parcelHelpers.export(exports, "fold_until", ()=>fold_until);
/**
 * Finds the first element in a given list for which the given function returns
 * `True`.
 *
 * Returns `Error(Nil)` if no such element is found.
 *
 * ## Examples
 *
 * ```gleam
 * assert find([1, 2, 3], fn(x) { x > 2 }) == Ok(3)
 * ```
 *
 * ```gleam
 * assert find([1, 2, 3], fn(x) { x > 4 }) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert find([], fn(_) { True }) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "find", ()=>find);
/**
 * Finds the first element in a given list for which the given function returns
 * `Ok(new_value)`, then returns the wrapped `new_value`.
 *
 * Returns `Error(Nil)` if no such element is found.
 *
 * ## Examples
 *
 * ```gleam
 * assert find_map([[], [2], [3]], first) == Ok(2)
 * ```
 *
 * ```gleam
 * assert find_map([[], []], first) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert find_map([], first) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "find_map", ()=>find_map);
/**
 * Returns `True` if the given function returns `True` for all the elements in
 * the given list. If the function returns `False` for any of the elements it
 * immediately returns `False` without checking the rest of the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert all([], fn(x) { x > 3 })
 * ```
 *
 * ```gleam
 * assert all([4, 5], fn(x) { x > 3 })
 * ```
 *
 * ```gleam
 * assert !all([4, 3], fn(x) { x > 3 })
 * ```
 */ parcelHelpers.export(exports, "all", ()=>all);
/**
 * Returns `True` if the given function returns `True` for any the elements in
 * the given list. If the function returns `True` for any of the elements it
 * immediately returns `True` without checking the rest of the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert !any([], fn(x) { x > 3 })
 * ```
 *
 * ```gleam
 * assert any([4, 5], fn(x) { x > 3 })
 * ```
 *
 * ```gleam
 * assert any([4, 3], fn(x) { x > 4 })
 * ```
 *
 * ```gleam
 * assert any([3, 4], fn(x) { x > 3 })
 * ```
 */ parcelHelpers.export(exports, "any", ()=>any);
/**
 * Takes two lists and returns a single list of 2-element tuples.
 *
 * If one of the lists is longer than the other, the remaining elements from
 * the longer list are not used.
 *
 * ## Examples
 *
 * ```gleam
 * assert zip([], []) == []
 * ```
 *
 * ```gleam
 * assert zip([1, 2], [3]) == [#(1, 3)]
 * ```
 *
 * ```gleam
 * assert zip([1], [3, 4]) == [#(1, 3)]
 * ```
 *
 * ```gleam
 * assert zip([1, 2], [3, 4]) == [#(1, 3), #(2, 4)]
 * ```
 */ parcelHelpers.export(exports, "zip", ()=>zip);
/**
 * Takes two lists and returns a single list of 2-element tuples.
 *
 * If one of the lists is longer than the other, an `Error` is returned.
 *
 * ## Examples
 *
 * ```gleam
 * assert strict_zip([], []) == Ok([])
 * ```
 *
 * ```gleam
 * assert strict_zip([1, 2], [3]) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert strict_zip([1], [3, 4]) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert strict_zip([1, 2], [3, 4]) == Ok([#(1, 3), #(2, 4)])
 * ```
 */ parcelHelpers.export(exports, "strict_zip", ()=>strict_zip);
/**
 * Takes a single list of 2-element tuples and returns two lists.
 *
 * ## Examples
 *
 * ```gleam
 * assert unzip([#(1, 2), #(3, 4)]) == #([1, 3], [2, 4])
 * ```
 *
 * ```gleam
 * assert unzip([]) == #([], [])
 * ```
 */ parcelHelpers.export(exports, "unzip", ()=>unzip);
/**
 * Inserts a given value between each existing element in a given list.
 *
 * This function runs in linear time and copies the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert intersperse([1, 1, 1], 2) == [1, 2, 1, 2, 1]
 * ```
 *
 * ```gleam
 * assert intersperse([], 2) == []
 * ```
 */ parcelHelpers.export(exports, "intersperse", ()=>intersperse);
/**
 * Removes any duplicate elements from a given list.
 *
 * This function returns in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert unique([1, 1, 1, 4, 7, 3, 3, 4]) == [1, 4, 7, 3]
 * ```
 */ parcelHelpers.export(exports, "unique", ()=>unique);
/**
 * Sorts from smallest to largest based upon the ordering specified by a given
 * function.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 *
 * assert sort([4, 3, 6, 5, 4, 1, 2], by: int.compare) == [1, 2, 3, 4, 4, 5, 6]
 * ```
 */ parcelHelpers.export(exports, "sort", ()=>sort);
parcelHelpers.export(exports, "range", ()=>range);
/**
 * Builds a list of a given value a given number of times.
 *
 * ## Examples
 *
 * ```gleam
 * assert repeat("a", times: 0) == []
 * ```
 *
 * ```gleam
 * assert repeat("a", times: 5) == ["a", "a", "a", "a", "a"]
 * ```
 */ parcelHelpers.export(exports, "repeat", ()=>repeat);
/**
 * Splits a list in two before the given index.
 *
 * If the list is not long enough to have the given index the before list will
 * be the input list, and the after list will be empty.
 *
 * ## Examples
 *
 * ```gleam
 * assert split([6, 7, 8, 9], 0) == #([], [6, 7, 8, 9])
 * ```
 *
 * ```gleam
 * assert split([6, 7, 8, 9], 2) == #([6, 7], [8, 9])
 * ```
 *
 * ```gleam
 * assert split([6, 7, 8, 9], 4) == #([6, 7, 8, 9], [])
 * ```
 */ parcelHelpers.export(exports, "split", ()=>split);
/**
 * Splits a list in two before the first element that a given function returns
 * `False` for.
 *
 * If the function returns `True` for all elements the first list will be the
 * input list, and the second list will be empty.
 *
 * ## Examples
 *
 * ```gleam
 * assert split_while([1, 2, 3, 4, 5], fn(x) { x <= 3 })
 *   == #([1, 2, 3], [4, 5])
 * ```
 *
 * ```gleam
 * assert split_while([1, 2, 3, 4, 5], fn(x) { x <= 5 })
 *   == #([1, 2, 3, 4, 5], [])
 * ```
 */ parcelHelpers.export(exports, "split_while", ()=>split_while);
/**
 * Given a list of 2-element tuples, finds the first tuple that has a given
 * key as the first element and returns the second element.
 *
 * If no tuple is found with the given key then `Error(Nil)` is returned.
 *
 * This function may be useful for interacting with Erlang code where lists of
 * tuples are common.
 *
 * ## Examples
 *
 * ```gleam
 * assert key_find([#("a", 0), #("b", 1)], "a") == Ok(0)
 * ```
 *
 * ```gleam
 * assert key_find([#("a", 0), #("b", 1)], "b") == Ok(1)
 * ```
 *
 * ```gleam
 * assert key_find([#("a", 0), #("b", 1)], "c") == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "key_find", ()=>key_find);
/**
 * Given a list of 2-element tuples, finds all tuples that have a given
 * key as the first element and returns the second element.
 *
 * This function may be useful for interacting with Erlang code where lists of
 * tuples are common.
 *
 * ## Examples
 *
 * ```gleam
 * assert key_filter([#("a", 0), #("b", 1), #("a", 2)], "a") == [0, 2]
 * ```
 *
 * ```gleam
 * assert key_filter([#("a", 0), #("b", 1)], "c") == []
 * ```
 */ parcelHelpers.export(exports, "key_filter", ()=>key_filter);
/**
 * Given a list of 2-element tuples, finds the first tuple that has a given
 * key as the first element. This function will return the second element
 * of the found tuple and list with tuple removed.
 *
 * If no tuple is found with the given key then `Error(Nil)` is returned.
 *
 * ## Examples
 *
 * ```gleam
 * assert key_pop([#("a", 0), #("b", 1)], "a") == Ok(#(0, [#("b", 1)]))
 * ```
 *
 * ```gleam
 * assert key_pop([#("a", 0), #("b", 1)], "b") == Ok(#(1, [#("a", 0)]))
 * ```
 *
 * ```gleam
 * assert key_pop([#("a", 0), #("b", 1)], "c") == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "key_pop", ()=>key_pop);
/**
 * Given a list of 2-element tuples, inserts a key and value into the list.
 *
 * If there was already a tuple with the key then it is replaced, otherwise it
 * is added to the end of the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert key_set([#(5, 0), #(4, 1)], 4, 100) == [#(5, 0), #(4, 100)]
 * ```
 *
 * ```gleam
 * assert key_set([#(5, 0), #(4, 1)], 1, 100) == [#(5, 0), #(4, 1), #(1, 100)]
 * ```
 */ parcelHelpers.export(exports, "key_set", ()=>key_set);
/**
 * Calls a function for each element in a list, discarding the return value.
 *
 * Useful for calling a side effect for every item of a list.
 *
 * ```gleam
 * import gleam/io
 *
 * assert each(["1", "2", "3"], io.println) == Nil
 * // 1
 * // 2
 * // 3
 * ```
 */ parcelHelpers.export(exports, "each", ()=>each);
/**
 * Calls a `Result` returning function for each element in a list, discarding
 * the return value. If the function returns `Error` then the iteration is
 * stopped and the error is returned.
 *
 * Useful for calling a side effect for every item of a list.
 *
 * ## Examples
 *
 * ```gleam
 * assert
 *   try_each(
 *     over: [1, 2, 3],
 *     with: function_that_might_fail,
 *   )
 *   == Ok(Nil)
 * ```
 */ parcelHelpers.export(exports, "try_each", ()=>try_each);
/**
 * Partitions a list into a tuple/pair of lists
 * by a given categorisation function.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 *
 * assert [1, 2, 3, 4, 5] |> partition(int.is_odd) == #([1, 3, 5], [2, 4])
 * ```
 */ parcelHelpers.export(exports, "partition", ()=>partition);
/**
 * Returns a list of sliding windows.
 *
 * ## Examples
 *
 * ```gleam
 * assert window([1,2,3,4,5], 3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
 * ```
 *
 * ```gleam
 * assert window([1, 2], 4) == []
 * ```
 */ parcelHelpers.export(exports, "window", ()=>window);
/**
 * Returns a list of tuples containing two contiguous elements.
 *
 * ## Examples
 *
 * ```gleam
 * assert window_by_2([1,2,3,4]) == [#(1, 2), #(2, 3), #(3, 4)]
 * ```
 *
 * ```gleam
 * assert window_by_2([1]) == []
 * ```
 */ parcelHelpers.export(exports, "window_by_2", ()=>window_by_2);
/**
 * Drops the first elements in a given list for which the predicate function returns `True`.
 *
 * ## Examples
 *
 * ```gleam
 * assert drop_while([1, 2, 3, 4], fn (x) { x < 3 }) == [3, 4]
 * ```
 */ parcelHelpers.export(exports, "drop_while", ()=>drop_while);
/**
 * Takes the first elements in a given list for which the predicate function returns `True`.
 *
 * ## Examples
 *
 * ```gleam
 * assert take_while([1, 2, 3, 2, 4], fn (x) { x < 3 }) == [1, 2]
 * ```
 */ parcelHelpers.export(exports, "take_while", ()=>take_while);
/**
 * Returns a list of chunks in which
 * the return value of calling `f` on each element is the same.
 *
 * ## Examples
 *
 * ```gleam
 * assert [1, 2, 2, 3, 4, 4, 6, 7, 7] |> chunk(by: fn(n) { n % 2 })
 *   == [[1], [2, 2], [3], [4, 4, 6], [7, 7]]
 * ```
 */ parcelHelpers.export(exports, "chunk", ()=>chunk);
/**
 * Returns a list of chunks containing `count` elements each.
 *
 * If the last chunk does not have `count` elements, it is instead
 * a partial chunk, with less than `count` elements.
 *
 * For any `count` less than 1 this function behaves as if it was set to 1.
 *
 * ## Examples
 *
 * ```gleam
 * assert [1, 2, 3, 4, 5, 6] |> sized_chunk(into: 2)
 *   == [[1, 2], [3, 4], [5, 6]]
 * ```
 *
 * ```gleam
 * assert [1, 2, 3, 4, 5, 6, 7, 8] |> sized_chunk(into: 3)
 *   == [[1, 2, 3], [4, 5, 6], [7, 8]]
 * ```
 */ parcelHelpers.export(exports, "sized_chunk", ()=>sized_chunk);
/**
 * This function acts similar to fold, but does not take an initial state.
 * Instead, it starts from the first element in the list
 * and combines it with each subsequent element in turn using the given
 * function. The function is called as `fun(accumulator, current_element)`.
 *
 * Returns `Ok` to indicate a successful run, and `Error` if called on an
 * empty list.
 *
 * ## Examples
 *
 * ```gleam
 * assert [] |> reduce(fn(acc, x) { acc + x }) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert [1, 2, 3, 4, 5] |> reduce(fn(acc, x) { acc + x }) == Ok(15)
 * ```
 */ parcelHelpers.export(exports, "reduce", ()=>reduce);
/**
 * Similar to `fold`, but yields the state of the accumulator at each stage.
 *
 * ## Examples
 *
 * ```gleam
 * assert scan(over: [1, 2, 3], from: 100, with: fn(acc, i) { acc + i })
 *   == [101, 103, 106]
 * ```
 */ parcelHelpers.export(exports, "scan", ()=>scan);
/**
 * Returns the last element in the given list.
 *
 * Returns `Error(Nil)` if the list is empty.
 *
 * This function runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert last([]) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert last([1, 2, 3, 4, 5]) == Ok(5)
 * ```
 */ parcelHelpers.export(exports, "last", ()=>last);
/**
 * Return unique combinations of elements in the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert combinations([1, 2, 3], 2) == [[1, 2], [1, 3], [2, 3]]
 * ```
 *
 * ```gleam
 * assert combinations([1, 2, 3, 4], 3)
 *   == [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
 * ```
 */ parcelHelpers.export(exports, "combinations", ()=>combinations);
/**
 * Return unique pair combinations of elements in the list.
 *
 * ## Examples
 *
 * ```gleam
 * assert combination_pairs([1, 2, 3]) == [#(1, 2), #(1, 3), #(2, 3)]
 * ```
 */ parcelHelpers.export(exports, "combination_pairs", ()=>combination_pairs);
/**
 * Transpose rows and columns of the list of lists.
 *
 * Notice: This function is not tail recursive,
 * and thus may exceed stack size if called,
 * with large lists (on the JavaScript target).
 *
 * ## Examples
 *
 * ```gleam
 * assert transpose([[1, 2, 3], [101, 102, 103]])
 *   == [[1, 101], [2, 102], [3, 103]]
 * ```
 */ parcelHelpers.export(exports, "transpose", ()=>transpose);
/**
 * Make a list alternating the elements from the given lists
 *
 * ## Examples
 *
 * ```gleam
 * assert interleave([[1, 2], [101, 102], [201, 202]])
 *   == [1, 101, 201, 2, 102, 202]
 * ```
 */ parcelHelpers.export(exports, "interleave", ()=>interleave);
/**
 * Takes a list, randomly sorts all items and returns the shuffled list.
 *
 * This function uses `float.random` to decide the order of the elements.
 *
 * ## Example
 *
 * ```gleam
 * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |> shuffle
 * // -> [1, 6, 9, 10, 3, 8, 4, 2, 7, 5]
 * ```
 */ parcelHelpers.export(exports, "shuffle", ()=>shuffle);
/**
 * Takes a list and a comparator, and returns the maximum element in the list
 *
 * ## Examples
 *
 * ```gleam
 * assert [1, 2, 3, 4, 5] |> list.max(int.compare) == Ok(5)
 * ```
 *
 * ```gleam
 * assert ["a", "c", "b"] |> list.max(string.compare) == Ok("c")
 * ```
 */ parcelHelpers.export(exports, "max", ()=>max);
/**
 * Returns a random sample of up to n elements from a list using reservoir
 * sampling via [Algorithm L](https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L).
 * Returns an empty list if the sample size is less than or equal to 0.
 *
 * Order is not random, only selection is.
 *
 * ## Examples
 *
 * ```gleam
 * sample([1, 2, 3, 4, 5], 3)
 * // -> [2, 4, 5]  // A random sample of 3 items
 * ```
 */ parcelHelpers.export(exports, "sample", ()=>sample);
/**
 * Returns all the permutations of a list.
 *
 * ## Examples
 *
 * ```gleam
 * assert permutations([1, 2]) == [[1, 2], [2, 1]]
 * ```
 */ parcelHelpers.export(exports, "permutations", ()=>permutations);
var _gleamMjs = require("../gleam.mjs");
var _dictMjs = require("../gleam/dict.mjs");
var _floatMjs = require("../gleam/float.mjs");
var _intMjs = require("../gleam/int.mjs");
var _orderMjs = require("../gleam/order.mjs");
const FILEPATH = "src/gleam/list.gleam";
class Continue extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const ContinueOrStop$Continue = ($0)=>new Continue($0);
const ContinueOrStop$isContinue = (value)=>value instanceof Continue;
const ContinueOrStop$Continue$0 = (value)=>value[0];
class Stop extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const ContinueOrStop$Stop = ($0)=>new Stop($0);
const ContinueOrStop$isStop = (value)=>value instanceof Stop;
const ContinueOrStop$Stop$0 = (value)=>value[0];
class Ascending extends (0, _gleamMjs.CustomType) {
}
class Descending extends (0, _gleamMjs.CustomType) {
}
const min_positive = 2.2250738585072014e-308;
function length_loop(loop$list, loop$count) {
    while(true){
        let list = loop$list;
        let count = loop$count;
        if (list instanceof (0, _gleamMjs.Empty)) return count;
        else {
            let list$1 = list.tail;
            loop$list = list$1;
            loop$count = count + 1;
        }
    }
}
function length(list) {
    return length_loop(list, 0);
}
function count_loop(loop$list, loop$predicate, loop$acc) {
    while(true){
        let list = loop$list;
        let predicate = loop$predicate;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return acc;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = predicate(first$1);
            if ($) {
                loop$list = rest$1;
                loop$predicate = predicate;
                loop$acc = acc + 1;
            } else {
                loop$list = rest$1;
                loop$predicate = predicate;
                loop$acc = acc;
            }
        }
    }
}
function count(list, predicate) {
    return count_loop(list, predicate, 0);
}
/**
 * Reverses a list and prepends it to another list.
 * This function runs in linear time, proportional to the length of the list
 * to prepend.
 * 
 * @ignore
 */ function reverse_and_prepend(loop$prefix, loop$suffix) {
    while(true){
        let prefix = loop$prefix;
        let suffix = loop$suffix;
        if (prefix instanceof (0, _gleamMjs.Empty)) return suffix;
        else {
            let first$1 = prefix.head;
            let rest$1 = prefix.tail;
            loop$prefix = rest$1;
            loop$suffix = (0, _gleamMjs.prepend)(first$1, suffix);
        }
    }
}
function reverse(list) {
    return reverse_and_prepend(list, (0, _gleamMjs.toList)([]));
}
function is_empty(list) {
    return (0, _gleamMjs.isEqual)(list, (0, _gleamMjs.toList)([]));
}
function contains(loop$list, loop$elem) {
    while(true){
        let list = loop$list;
        let elem = loop$elem;
        if (list instanceof (0, _gleamMjs.Empty)) return false;
        else {
            let first$1 = list.head;
            if ((0, _gleamMjs.isEqual)(first$1, elem)) return true;
            else {
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$elem = elem;
            }
        }
    }
}
function first(list) {
    if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
    else {
        let first$1 = list.head;
        return new (0, _gleamMjs.Ok)(first$1);
    }
}
function rest(list) {
    if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
    else {
        let rest$1 = list.tail;
        return new (0, _gleamMjs.Ok)(rest$1);
    }
}
function group(list, key) {
    return _dictMjs.group(key, list);
}
function filter_loop(loop$list, loop$fun, loop$acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let _block;
            let $ = fun(first$1);
            if ($) _block = (0, _gleamMjs.prepend)(first$1, acc);
            else _block = acc;
            let new_acc = _block;
            loop$list = rest$1;
            loop$fun = fun;
            loop$acc = new_acc;
        }
    }
}
function filter(list, predicate) {
    return filter_loop(list, predicate, (0, _gleamMjs.toList)([]));
}
function filter_map_loop(loop$list, loop$fun, loop$acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let _block;
            let $ = fun(first$1);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let first$2 = $[0];
                _block = (0, _gleamMjs.prepend)(first$2, acc);
            } else _block = acc;
            let new_acc = _block;
            loop$list = rest$1;
            loop$fun = fun;
            loop$acc = new_acc;
        }
    }
}
function filter_map(list, fun) {
    return filter_map_loop(list, fun, (0, _gleamMjs.toList)([]));
}
function map_loop(loop$list, loop$fun, loop$acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            loop$list = rest$1;
            loop$fun = fun;
            loop$acc = (0, _gleamMjs.prepend)(fun(first$1), acc);
        }
    }
}
function map(list, fun) {
    return map_loop(list, fun, (0, _gleamMjs.toList)([]));
}
function map2_loop(loop$list1, loop$list2, loop$fun, loop$acc) {
    while(true){
        let list1 = loop$list1;
        let list2 = loop$list2;
        let fun = loop$fun;
        let acc = loop$acc;
        if (list1 instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else if (list2 instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let a = list1.head;
            let as_ = list1.tail;
            let b = list2.head;
            let bs = list2.tail;
            loop$list1 = as_;
            loop$list2 = bs;
            loop$fun = fun;
            loop$acc = (0, _gleamMjs.prepend)(fun(a, b), acc);
        }
    }
}
function map2(list1, list2, fun) {
    return map2_loop(list1, list2, fun, (0, _gleamMjs.toList)([]));
}
function map_fold_loop(loop$list, loop$fun, loop$acc, loop$list_acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let acc = loop$acc;
        let list_acc = loop$list_acc;
        if (list instanceof (0, _gleamMjs.Empty)) return [
            acc,
            reverse(list_acc)
        ];
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(acc, first$1);
            let acc$1;
            let first$2;
            acc$1 = $[0];
            first$2 = $[1];
            loop$list = rest$1;
            loop$fun = fun;
            loop$acc = acc$1;
            loop$list_acc = (0, _gleamMjs.prepend)(first$2, list_acc);
        }
    }
}
function map_fold(list, initial, fun) {
    return map_fold_loop(list, fun, initial, (0, _gleamMjs.toList)([]));
}
function index_map_loop(loop$list, loop$fun, loop$index, loop$acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let index = loop$index;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let acc$1 = (0, _gleamMjs.prepend)(fun(first$1, index), acc);
            loop$list = rest$1;
            loop$fun = fun;
            loop$index = index + 1;
            loop$acc = acc$1;
        }
    }
}
function index_map(list, fun) {
    return index_map_loop(list, fun, 0, (0, _gleamMjs.toList)([]));
}
function try_map_loop(loop$list, loop$fun, loop$acc) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Ok)(reverse(acc));
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(first$1);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let first$2 = $[0];
                loop$list = rest$1;
                loop$fun = fun;
                loop$acc = (0, _gleamMjs.prepend)(first$2, acc);
            } else return $;
        }
    }
}
function try_map(list, fun) {
    return try_map_loop(list, fun, (0, _gleamMjs.toList)([]));
}
function drop(loop$list, loop$n) {
    while(true){
        let list = loop$list;
        let n = loop$n;
        let $ = n <= 0;
        if ($) return list;
        else {
            if (list instanceof (0, _gleamMjs.Empty)) return list;
            else {
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$n = n - 1;
            }
        }
    }
}
function take_loop(loop$list, loop$n, loop$acc) {
    while(true){
        let list = loop$list;
        let n = loop$n;
        let acc = loop$acc;
        let $ = n <= 0;
        if ($) return reverse(acc);
        else {
            if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
            else {
                let first$1 = list.head;
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$n = n - 1;
                loop$acc = (0, _gleamMjs.prepend)(first$1, acc);
            }
        }
    }
}
function take(list, n) {
    return take_loop(list, n, (0, _gleamMjs.toList)([]));
}
function new$() {
    return (0, _gleamMjs.toList)([]);
}
function wrap(item) {
    return (0, _gleamMjs.toList)([
        item
    ]);
}
function append_loop(loop$first, loop$second) {
    while(true){
        let first = loop$first;
        let second = loop$second;
        if (first instanceof (0, _gleamMjs.Empty)) return second;
        else {
            let first$1 = first.head;
            let rest$1 = first.tail;
            loop$first = rest$1;
            loop$second = (0, _gleamMjs.prepend)(first$1, second);
        }
    }
}
function append(first, second) {
    return append_loop(reverse(first), second);
}
function prepend(list, item) {
    return (0, _gleamMjs.prepend)(item, list);
}
function flatten_loop(loop$lists, loop$acc) {
    while(true){
        let lists = loop$lists;
        let acc = loop$acc;
        if (lists instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let list = lists.head;
            let further_lists = lists.tail;
            loop$lists = further_lists;
            loop$acc = reverse_and_prepend(list, acc);
        }
    }
}
function flatten(lists) {
    return flatten_loop(lists, (0, _gleamMjs.toList)([]));
}
function flat_map(list, fun) {
    return flatten(map(list, fun));
}
function fold(loop$list, loop$initial, loop$fun) {
    while(true){
        let list = loop$list;
        let initial = loop$initial;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            loop$list = rest$1;
            loop$initial = fun(initial, first$1);
            loop$fun = fun;
        }
    }
}
function fold_right(list, initial, fun) {
    if (list instanceof (0, _gleamMjs.Empty)) return initial;
    else {
        let first$1 = list.head;
        let rest$1 = list.tail;
        return fun(fold_right(rest$1, initial, fun), first$1);
    }
}
function index_fold_loop(loop$over, loop$acc, loop$with, loop$index) {
    while(true){
        let over = loop$over;
        let acc = loop$acc;
        let with$ = loop$with;
        let index = loop$index;
        if (over instanceof (0, _gleamMjs.Empty)) return acc;
        else {
            let first$1 = over.head;
            let rest$1 = over.tail;
            loop$over = rest$1;
            loop$acc = with$(acc, first$1, index);
            loop$with = with$;
            loop$index = index + 1;
        }
    }
}
function index_fold(list, initial, fun) {
    return index_fold_loop(list, initial, fun, 0);
}
function try_fold(loop$list, loop$initial, loop$fun) {
    while(true){
        let list = loop$list;
        let initial = loop$initial;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Ok)(initial);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(initial, first$1);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let result = $[0];
                loop$list = rest$1;
                loop$initial = result;
                loop$fun = fun;
            } else return $;
        }
    }
}
function fold_until(loop$list, loop$initial, loop$fun) {
    while(true){
        let list = loop$list;
        let initial = loop$initial;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(initial, first$1);
            if ($ instanceof Continue) {
                let next_accumulator = $[0];
                loop$list = rest$1;
                loop$initial = next_accumulator;
                loop$fun = fun;
            } else {
                let b = $[0];
                return b;
            }
        }
    }
}
function find(loop$list, loop$is_desired) {
    while(true){
        let list = loop$list;
        let is_desired = loop$is_desired;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = is_desired(first$1);
            if ($) return new (0, _gleamMjs.Ok)(first$1);
            else {
                loop$list = rest$1;
                loop$is_desired = is_desired;
            }
        }
    }
}
function find_map(loop$list, loop$fun) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(first$1);
            if ($ instanceof (0, _gleamMjs.Ok)) return $;
            else {
                loop$list = rest$1;
                loop$fun = fun;
            }
        }
    }
}
function all(loop$list, loop$predicate) {
    while(true){
        let list = loop$list;
        let predicate = loop$predicate;
        if (list instanceof (0, _gleamMjs.Empty)) return true;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = predicate(first$1);
            if ($) {
                loop$list = rest$1;
                loop$predicate = predicate;
            } else return $;
        }
    }
}
function any(loop$list, loop$predicate) {
    while(true){
        let list = loop$list;
        let predicate = loop$predicate;
        if (list instanceof (0, _gleamMjs.Empty)) return false;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = predicate(first$1);
            if ($) return $;
            else {
                loop$list = rest$1;
                loop$predicate = predicate;
            }
        }
    }
}
function zip_loop(loop$one, loop$other, loop$acc) {
    while(true){
        let one = loop$one;
        let other = loop$other;
        let acc = loop$acc;
        if (one instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else if (other instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first_one = one.head;
            let rest_one = one.tail;
            let first_other = other.head;
            let rest_other = other.tail;
            loop$one = rest_one;
            loop$other = rest_other;
            loop$acc = (0, _gleamMjs.prepend)([
                first_one,
                first_other
            ], acc);
        }
    }
}
function zip(list, other) {
    return zip_loop(list, other, (0, _gleamMjs.toList)([]));
}
function strict_zip_loop(loop$one, loop$other, loop$acc) {
    while(true){
        let one = loop$one;
        let other = loop$other;
        let acc = loop$acc;
        if (one instanceof (0, _gleamMjs.Empty)) {
            if (other instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Ok)(reverse(acc));
            else return new (0, _gleamMjs.Error)(undefined);
        } else if (other instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
        else {
            let first_one = one.head;
            let rest_one = one.tail;
            let first_other = other.head;
            let rest_other = other.tail;
            loop$one = rest_one;
            loop$other = rest_other;
            loop$acc = (0, _gleamMjs.prepend)([
                first_one,
                first_other
            ], acc);
        }
    }
}
function strict_zip(list, other) {
    return strict_zip_loop(list, other, (0, _gleamMjs.toList)([]));
}
function unzip_loop(loop$input, loop$one, loop$other) {
    while(true){
        let input = loop$input;
        let one = loop$one;
        let other = loop$other;
        if (input instanceof (0, _gleamMjs.Empty)) return [
            reverse(one),
            reverse(other)
        ];
        else {
            let rest$1 = input.tail;
            let first_one = input.head[0];
            let first_other = input.head[1];
            loop$input = rest$1;
            loop$one = (0, _gleamMjs.prepend)(first_one, one);
            loop$other = (0, _gleamMjs.prepend)(first_other, other);
        }
    }
}
function unzip(input) {
    return unzip_loop(input, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
}
function intersperse_loop(loop$list, loop$separator, loop$acc) {
    while(true){
        let list = loop$list;
        let separator = loop$separator;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            loop$list = rest$1;
            loop$separator = separator;
            loop$acc = (0, _gleamMjs.prepend)(first$1, (0, _gleamMjs.prepend)(separator, acc));
        }
    }
}
function intersperse(list, elem) {
    if (list instanceof (0, _gleamMjs.Empty)) return list;
    else {
        let $ = list.tail;
        if ($ instanceof (0, _gleamMjs.Empty)) return list;
        else {
            let first$1 = list.head;
            let rest$1 = $;
            return intersperse_loop(rest$1, elem, (0, _gleamMjs.toList)([
                first$1
            ]));
        }
    }
}
function unique_loop(loop$list, loop$seen, loop$acc) {
    while(true){
        let list = loop$list;
        let seen = loop$seen;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = _dictMjs.has_key(seen, first$1);
            if ($) {
                loop$list = rest$1;
                loop$seen = seen;
                loop$acc = acc;
            } else {
                loop$list = rest$1;
                loop$seen = _dictMjs.insert(seen, first$1, undefined);
                loop$acc = (0, _gleamMjs.prepend)(first$1, acc);
            }
        }
    }
}
function unique(list) {
    return unique_loop(list, _dictMjs.new$(), (0, _gleamMjs.toList)([]));
}
/**
 * Given a list it returns slices of it that are locally sorted in ascending
 * order.
 *
 * Imagine you have this list:
 *
 * ```
 *   [1, 2, 3, 2, 1, 0]
 *    ^^^^^^^  ^^^^^^^ This is a slice in descending order
 *    |
 *    | This is a slice that is sorted in ascending order
 * ```
 *
 * So the produced result will contain these two slices, each one sorted in
 * ascending order: `[[1, 2, 3], [0, 1, 2]]`.
 *
 * - `growing` is an accumulator with the current slice being grown
 * - `direction` is the growing direction of the slice being grown, it could
 *   either be ascending or strictly descending
 * - `prev` is the previous element that needs to be added to the growing slice
 *   it is carried around to check whether we have to keep growing the current
 *   slice or not
 * - `acc` is the accumulator containing the slices sorted in ascending order
 * 
 * @ignore
 */ function sequences(loop$list, loop$compare, loop$growing, loop$direction, loop$prev, loop$acc) {
    while(true){
        let list = loop$list;
        let compare = loop$compare;
        let growing = loop$growing;
        let direction = loop$direction;
        let prev = loop$prev;
        let acc = loop$acc;
        let growing$1 = (0, _gleamMjs.prepend)(prev, growing);
        if (list instanceof (0, _gleamMjs.Empty)) {
            if (direction instanceof Ascending) return (0, _gleamMjs.prepend)(reverse(growing$1), acc);
            else return (0, _gleamMjs.prepend)(growing$1, acc);
        } else {
            let new$1 = list.head;
            let rest$1 = list.tail;
            let $ = compare(prev, new$1);
            if (direction instanceof Ascending) {
                if ($ instanceof _orderMjs.Lt) {
                    loop$list = rest$1;
                    loop$compare = compare;
                    loop$growing = growing$1;
                    loop$direction = direction;
                    loop$prev = new$1;
                    loop$acc = acc;
                } else if ($ instanceof _orderMjs.Eq) {
                    loop$list = rest$1;
                    loop$compare = compare;
                    loop$growing = growing$1;
                    loop$direction = direction;
                    loop$prev = new$1;
                    loop$acc = acc;
                } else {
                    let _block;
                    if (direction instanceof Ascending) _block = (0, _gleamMjs.prepend)(reverse(growing$1), acc);
                    else _block = (0, _gleamMjs.prepend)(growing$1, acc);
                    let acc$1 = _block;
                    if (rest$1 instanceof (0, _gleamMjs.Empty)) return (0, _gleamMjs.prepend)((0, _gleamMjs.toList)([
                        new$1
                    ]), acc$1);
                    else {
                        let next = rest$1.head;
                        let rest$2 = rest$1.tail;
                        let _block$1;
                        let $1 = compare(new$1, next);
                        if ($1 instanceof _orderMjs.Lt) _block$1 = new Ascending();
                        else if ($1 instanceof _orderMjs.Eq) _block$1 = new Ascending();
                        else _block$1 = new Descending();
                        let direction$1 = _block$1;
                        loop$list = rest$2;
                        loop$compare = compare;
                        loop$growing = (0, _gleamMjs.toList)([
                            new$1
                        ]);
                        loop$direction = direction$1;
                        loop$prev = next;
                        loop$acc = acc$1;
                    }
                }
            } else if ($ instanceof _orderMjs.Lt) {
                let _block;
                if (direction instanceof Ascending) _block = (0, _gleamMjs.prepend)(reverse(growing$1), acc);
                else _block = (0, _gleamMjs.prepend)(growing$1, acc);
                let acc$1 = _block;
                if (rest$1 instanceof (0, _gleamMjs.Empty)) return (0, _gleamMjs.prepend)((0, _gleamMjs.toList)([
                    new$1
                ]), acc$1);
                else {
                    let next = rest$1.head;
                    let rest$2 = rest$1.tail;
                    let _block$1;
                    let $1 = compare(new$1, next);
                    if ($1 instanceof _orderMjs.Lt) _block$1 = new Ascending();
                    else if ($1 instanceof _orderMjs.Eq) _block$1 = new Ascending();
                    else _block$1 = new Descending();
                    let direction$1 = _block$1;
                    loop$list = rest$2;
                    loop$compare = compare;
                    loop$growing = (0, _gleamMjs.toList)([
                        new$1
                    ]);
                    loop$direction = direction$1;
                    loop$prev = next;
                    loop$acc = acc$1;
                }
            } else if ($ instanceof _orderMjs.Eq) {
                let _block;
                if (direction instanceof Ascending) _block = (0, _gleamMjs.prepend)(reverse(growing$1), acc);
                else _block = (0, _gleamMjs.prepend)(growing$1, acc);
                let acc$1 = _block;
                if (rest$1 instanceof (0, _gleamMjs.Empty)) return (0, _gleamMjs.prepend)((0, _gleamMjs.toList)([
                    new$1
                ]), acc$1);
                else {
                    let next = rest$1.head;
                    let rest$2 = rest$1.tail;
                    let _block$1;
                    let $1 = compare(new$1, next);
                    if ($1 instanceof _orderMjs.Lt) _block$1 = new Ascending();
                    else if ($1 instanceof _orderMjs.Eq) _block$1 = new Ascending();
                    else _block$1 = new Descending();
                    let direction$1 = _block$1;
                    loop$list = rest$2;
                    loop$compare = compare;
                    loop$growing = (0, _gleamMjs.toList)([
                        new$1
                    ]);
                    loop$direction = direction$1;
                    loop$prev = next;
                    loop$acc = acc$1;
                }
            } else {
                loop$list = rest$1;
                loop$compare = compare;
                loop$growing = growing$1;
                loop$direction = direction;
                loop$prev = new$1;
                loop$acc = acc;
            }
        }
    }
}
/**
 * Merges two lists sorted in ascending order into a single list sorted in
 * descending order according to the given comparator function.
 *
 * This reversing of the sort order is not avoidable if we want to implement
 * merge as a tail recursive function. We could reverse the accumulator before
 * returning it but that would end up being less efficient; so the merging
 * algorithm has to play around this.
 * 
 * @ignore
 */ function merge_ascendings(loop$list1, loop$list2, loop$compare, loop$acc) {
    while(true){
        let list1 = loop$list1;
        let list2 = loop$list2;
        let compare = loop$compare;
        let acc = loop$acc;
        if (list1 instanceof (0, _gleamMjs.Empty)) {
            let list = list2;
            return reverse_and_prepend(list, acc);
        } else if (list2 instanceof (0, _gleamMjs.Empty)) {
            let list = list1;
            return reverse_and_prepend(list, acc);
        } else {
            let first1 = list1.head;
            let rest1 = list1.tail;
            let first2 = list2.head;
            let rest2 = list2.tail;
            let $ = compare(first1, first2);
            if ($ instanceof _orderMjs.Lt) {
                loop$list1 = rest1;
                loop$list2 = list2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first1, acc);
            } else if ($ instanceof _orderMjs.Eq) {
                loop$list1 = list1;
                loop$list2 = rest2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first2, acc);
            } else {
                loop$list1 = list1;
                loop$list2 = rest2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first2, acc);
            }
        }
    }
}
/**
 * Given a list of ascending lists, it merges adjacent pairs into a single
 * descending list, halving their number.
 * It returns a list of the remaining descending lists.
 * 
 * @ignore
 */ function merge_ascending_pairs(loop$sequences, loop$compare, loop$acc) {
    while(true){
        let sequences = loop$sequences;
        let compare = loop$compare;
        let acc = loop$acc;
        if (sequences instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let $ = sequences.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let sequence = sequences.head;
                return reverse((0, _gleamMjs.prepend)(reverse(sequence), acc));
            } else {
                let ascending1 = sequences.head;
                let ascending2 = $.head;
                let rest$1 = $.tail;
                let descending = merge_ascendings(ascending1, ascending2, compare, (0, _gleamMjs.toList)([]));
                loop$sequences = rest$1;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(descending, acc);
            }
        }
    }
}
/**
 * This is exactly the same as merge_ascendings but mirrored: it merges two
 * lists sorted in descending order into a single list sorted in ascending
 * order according to the given comparator function.
 *
 * This reversing of the sort order is not avoidable if we want to implement
 * merge as a tail recursive function. We could reverse the accumulator before
 * returning it but that would end up being less efficient; so the merging
 * algorithm has to play around this.
 * 
 * @ignore
 */ function merge_descendings(loop$list1, loop$list2, loop$compare, loop$acc) {
    while(true){
        let list1 = loop$list1;
        let list2 = loop$list2;
        let compare = loop$compare;
        let acc = loop$acc;
        if (list1 instanceof (0, _gleamMjs.Empty)) {
            let list = list2;
            return reverse_and_prepend(list, acc);
        } else if (list2 instanceof (0, _gleamMjs.Empty)) {
            let list = list1;
            return reverse_and_prepend(list, acc);
        } else {
            let first1 = list1.head;
            let rest1 = list1.tail;
            let first2 = list2.head;
            let rest2 = list2.tail;
            let $ = compare(first1, first2);
            if ($ instanceof _orderMjs.Lt) {
                loop$list1 = list1;
                loop$list2 = rest2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first2, acc);
            } else if ($ instanceof _orderMjs.Eq) {
                loop$list1 = rest1;
                loop$list2 = list2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first1, acc);
            } else {
                loop$list1 = rest1;
                loop$list2 = list2;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(first1, acc);
            }
        }
    }
}
/**
 * This is the same as merge_ascending_pairs but flipped for descending lists.
 * 
 * @ignore
 */ function merge_descending_pairs(loop$sequences, loop$compare, loop$acc) {
    while(true){
        let sequences = loop$sequences;
        let compare = loop$compare;
        let acc = loop$acc;
        if (sequences instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let $ = sequences.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let sequence = sequences.head;
                return reverse((0, _gleamMjs.prepend)(reverse(sequence), acc));
            } else {
                let descending1 = sequences.head;
                let descending2 = $.head;
                let rest$1 = $.tail;
                let ascending = merge_descendings(descending1, descending2, compare, (0, _gleamMjs.toList)([]));
                loop$sequences = rest$1;
                loop$compare = compare;
                loop$acc = (0, _gleamMjs.prepend)(ascending, acc);
            }
        }
    }
}
/**
 * Given some some sorted sequences (assumed to be sorted in `direction`) it
 * merges them all together until we're left with just a list sorted in
 * ascending order.
 * 
 * @ignore
 */ function merge_all(loop$sequences, loop$direction, loop$compare) {
    while(true){
        let sequences = loop$sequences;
        let direction = loop$direction;
        let compare = loop$compare;
        if (sequences instanceof (0, _gleamMjs.Empty)) return sequences;
        else if (direction instanceof Ascending) {
            let $ = sequences.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let sequence = sequences.head;
                return sequence;
            } else {
                let sequences$1 = merge_ascending_pairs(sequences, compare, (0, _gleamMjs.toList)([]));
                loop$sequences = sequences$1;
                loop$direction = new Descending();
                loop$compare = compare;
            }
        } else {
            let $ = sequences.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let sequence = sequences.head;
                return reverse(sequence);
            } else {
                let sequences$1 = merge_descending_pairs(sequences, compare, (0, _gleamMjs.toList)([]));
                loop$sequences = sequences$1;
                loop$direction = new Ascending();
                loop$compare = compare;
            }
        }
    }
}
function sort(list, compare) {
    if (list instanceof (0, _gleamMjs.Empty)) return list;
    else {
        let $ = list.tail;
        if ($ instanceof (0, _gleamMjs.Empty)) return list;
        else {
            let x = list.head;
            let y = $.head;
            let rest$1 = $.tail;
            let _block;
            let $1 = compare(x, y);
            if ($1 instanceof _orderMjs.Lt) _block = new Ascending();
            else if ($1 instanceof _orderMjs.Eq) _block = new Ascending();
            else _block = new Descending();
            let direction = _block;
            let sequences$1 = sequences(rest$1, compare, (0, _gleamMjs.toList)([
                x
            ]), direction, y, (0, _gleamMjs.toList)([]));
            return merge_all(sequences$1, new Ascending(), compare);
        }
    }
}
function range_loop(loop$start, loop$stop, loop$acc) {
    while(true){
        let start = loop$start;
        let stop = loop$stop;
        let acc = loop$acc;
        let $ = _intMjs.compare(start, stop);
        if ($ instanceof _orderMjs.Lt) {
            loop$start = start;
            loop$stop = stop - 1;
            loop$acc = (0, _gleamMjs.prepend)(stop, acc);
        } else if ($ instanceof _orderMjs.Eq) return (0, _gleamMjs.prepend)(stop, acc);
        else {
            loop$start = start;
            loop$stop = stop + 1;
            loop$acc = (0, _gleamMjs.prepend)(stop, acc);
        }
    }
}
function range(start, stop) {
    return range_loop(start, stop, (0, _gleamMjs.toList)([]));
}
function repeat_loop(loop$item, loop$times, loop$acc) {
    while(true){
        let item = loop$item;
        let times = loop$times;
        let acc = loop$acc;
        let $ = times <= 0;
        if ($) return acc;
        else {
            loop$item = item;
            loop$times = times - 1;
            loop$acc = (0, _gleamMjs.prepend)(item, acc);
        }
    }
}
function repeat(a, times) {
    return repeat_loop(a, times, (0, _gleamMjs.toList)([]));
}
function split_loop(loop$list, loop$n, loop$taken) {
    while(true){
        let list = loop$list;
        let n = loop$n;
        let taken = loop$taken;
        let $ = n <= 0;
        if ($) return [
            reverse(taken),
            list
        ];
        else {
            if (list instanceof (0, _gleamMjs.Empty)) return [
                reverse(taken),
                (0, _gleamMjs.toList)([])
            ];
            else {
                let first$1 = list.head;
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$n = n - 1;
                loop$taken = (0, _gleamMjs.prepend)(first$1, taken);
            }
        }
    }
}
function split(list, index) {
    return split_loop(list, index, (0, _gleamMjs.toList)([]));
}
function split_while_loop(loop$list, loop$f, loop$acc) {
    while(true){
        let list = loop$list;
        let f = loop$f;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return [
            reverse(acc),
            (0, _gleamMjs.toList)([])
        ];
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = f(first$1);
            if ($) {
                loop$list = rest$1;
                loop$f = f;
                loop$acc = (0, _gleamMjs.prepend)(first$1, acc);
            } else return [
                reverse(acc),
                list
            ];
        }
    }
}
function split_while(list, predicate) {
    return split_while_loop(list, predicate, (0, _gleamMjs.toList)([]));
}
function key_find(keyword_list, desired_key) {
    return find_map(keyword_list, (keyword)=>{
        let key;
        let value;
        key = keyword[0];
        value = keyword[1];
        let $ = (0, _gleamMjs.isEqual)(key, desired_key);
        if ($) return new (0, _gleamMjs.Ok)(value);
        else return new (0, _gleamMjs.Error)(undefined);
    });
}
function key_filter(keyword_list, desired_key) {
    return filter_map(keyword_list, (keyword)=>{
        let key;
        let value;
        key = keyword[0];
        value = keyword[1];
        let $ = (0, _gleamMjs.isEqual)(key, desired_key);
        if ($) return new (0, _gleamMjs.Ok)(value);
        else return new (0, _gleamMjs.Error)(undefined);
    });
}
function key_pop_loop(loop$list, loop$key, loop$checked) {
    while(true){
        let list = loop$list;
        let key = loop$key;
        let checked = loop$checked;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
        else {
            let k = list.head[0];
            if ((0, _gleamMjs.isEqual)(k, key)) {
                let rest$1 = list.tail;
                let v = list.head[1];
                return new (0, _gleamMjs.Ok)([
                    v,
                    reverse_and_prepend(checked, rest$1)
                ]);
            } else {
                let first$1 = list.head;
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$key = key;
                loop$checked = (0, _gleamMjs.prepend)(first$1, checked);
            }
        }
    }
}
function key_pop(list, key) {
    return key_pop_loop(list, key, (0, _gleamMjs.toList)([]));
}
function key_set_loop(loop$list, loop$key, loop$value, loop$inspected) {
    while(true){
        let list = loop$list;
        let key = loop$key;
        let value = loop$value;
        let inspected = loop$inspected;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse((0, _gleamMjs.prepend)([
            key,
            value
        ], inspected));
        else {
            let k = list.head[0];
            if ((0, _gleamMjs.isEqual)(k, key)) {
                let rest$1 = list.tail;
                return reverse_and_prepend(inspected, (0, _gleamMjs.prepend)([
                    k,
                    value
                ], rest$1));
            } else {
                let first$1 = list.head;
                let rest$1 = list.tail;
                loop$list = rest$1;
                loop$key = key;
                loop$value = value;
                loop$inspected = (0, _gleamMjs.prepend)(first$1, inspected);
            }
        }
    }
}
function key_set(list, key, value) {
    return key_set_loop(list, key, value, (0, _gleamMjs.toList)([]));
}
function each(loop$list, loop$f) {
    while(true){
        let list = loop$list;
        let f = loop$f;
        if (list instanceof (0, _gleamMjs.Empty)) return undefined;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            f(first$1);
            loop$list = rest$1;
            loop$f = f;
        }
    }
}
function try_each(loop$list, loop$fun) {
    while(true){
        let list = loop$list;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Ok)(undefined);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = fun(first$1);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                loop$list = rest$1;
                loop$fun = fun;
            } else return $;
        }
    }
}
function partition_loop(loop$list, loop$categorise, loop$trues, loop$falses) {
    while(true){
        let list = loop$list;
        let categorise = loop$categorise;
        let trues = loop$trues;
        let falses = loop$falses;
        if (list instanceof (0, _gleamMjs.Empty)) return [
            reverse(trues),
            reverse(falses)
        ];
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = categorise(first$1);
            if ($) {
                loop$list = rest$1;
                loop$categorise = categorise;
                loop$trues = (0, _gleamMjs.prepend)(first$1, trues);
                loop$falses = falses;
            } else {
                loop$list = rest$1;
                loop$categorise = categorise;
                loop$trues = trues;
                loop$falses = (0, _gleamMjs.prepend)(first$1, falses);
            }
        }
    }
}
function partition(list, categorise) {
    return partition_loop(list, categorise, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
}
function window_loop(loop$acc, loop$list, loop$n) {
    while(true){
        let acc = loop$acc;
        let list = loop$list;
        let n = loop$n;
        let window$1 = take(list, n);
        let $ = length(window$1) === n;
        if ($) {
            loop$acc = (0, _gleamMjs.prepend)(window$1, acc);
            loop$list = drop(list, 1);
            loop$n = n;
        } else return reverse(acc);
    }
}
function window(list, n) {
    let $ = n <= 0;
    if ($) return (0, _gleamMjs.toList)([]);
    else return window_loop((0, _gleamMjs.toList)([]), list, n);
}
function window_by_2(list) {
    return zip(list, drop(list, 1));
}
function drop_while(loop$list, loop$predicate) {
    while(true){
        let list = loop$list;
        let predicate = loop$predicate;
        if (list instanceof (0, _gleamMjs.Empty)) return list;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = predicate(first$1);
            if ($) {
                loop$list = rest$1;
                loop$predicate = predicate;
            } else return (0, _gleamMjs.prepend)(first$1, rest$1);
        }
    }
}
function take_while_loop(loop$list, loop$predicate, loop$acc) {
    while(true){
        let list = loop$list;
        let predicate = loop$predicate;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = predicate(first$1);
            if ($) {
                loop$list = rest$1;
                loop$predicate = predicate;
                loop$acc = (0, _gleamMjs.prepend)(first$1, acc);
            } else return reverse(acc);
        }
    }
}
function take_while(list, predicate) {
    return take_while_loop(list, predicate, (0, _gleamMjs.toList)([]));
}
function chunk_loop(loop$list, loop$f, loop$previous_key, loop$current_chunk, loop$acc) {
    while(true){
        let list = loop$list;
        let f = loop$f;
        let previous_key = loop$previous_key;
        let current_chunk = loop$current_chunk;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse((0, _gleamMjs.prepend)(reverse(current_chunk), acc));
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let key = f(first$1);
            let $ = (0, _gleamMjs.isEqual)(key, previous_key);
            if ($) {
                loop$list = rest$1;
                loop$f = f;
                loop$previous_key = key;
                loop$current_chunk = (0, _gleamMjs.prepend)(first$1, current_chunk);
                loop$acc = acc;
            } else {
                let new_acc = (0, _gleamMjs.prepend)(reverse(current_chunk), acc);
                loop$list = rest$1;
                loop$f = f;
                loop$previous_key = key;
                loop$current_chunk = (0, _gleamMjs.toList)([
                    first$1
                ]);
                loop$acc = new_acc;
            }
        }
    }
}
function chunk(list, f) {
    if (list instanceof (0, _gleamMjs.Empty)) return list;
    else {
        let first$1 = list.head;
        let rest$1 = list.tail;
        return chunk_loop(rest$1, f, f(first$1), (0, _gleamMjs.toList)([
            first$1
        ]), (0, _gleamMjs.toList)([]));
    }
}
function sized_chunk_loop(loop$list, loop$count, loop$left, loop$current_chunk, loop$acc) {
    while(true){
        let list = loop$list;
        let count = loop$count;
        let left = loop$left;
        let current_chunk = loop$current_chunk;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) {
            if (current_chunk instanceof (0, _gleamMjs.Empty)) return reverse(acc);
            else {
                let remaining = current_chunk;
                return reverse((0, _gleamMjs.prepend)(reverse(remaining), acc));
            }
        } else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let chunk$1 = (0, _gleamMjs.prepend)(first$1, current_chunk);
            let $ = left > 1;
            if ($) {
                loop$list = rest$1;
                loop$count = count;
                loop$left = left - 1;
                loop$current_chunk = chunk$1;
                loop$acc = acc;
            } else {
                loop$list = rest$1;
                loop$count = count;
                loop$left = count;
                loop$current_chunk = (0, _gleamMjs.toList)([]);
                loop$acc = (0, _gleamMjs.prepend)(reverse(chunk$1), acc);
            }
        }
    }
}
function sized_chunk(list, count) {
    return sized_chunk_loop(list, count, count, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
}
function reduce(list, fun) {
    if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
    else {
        let first$1 = list.head;
        let rest$1 = list.tail;
        return new (0, _gleamMjs.Ok)(fold(rest$1, first$1, fun));
    }
}
function scan_loop(loop$list, loop$accumulator, loop$accumulated, loop$fun) {
    while(true){
        let list = loop$list;
        let accumulator = loop$accumulator;
        let accumulated = loop$accumulated;
        let fun = loop$fun;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(accumulated);
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let next = fun(accumulator, first$1);
            loop$list = rest$1;
            loop$accumulator = next;
            loop$accumulated = (0, _gleamMjs.prepend)(next, accumulated);
            loop$fun = fun;
        }
    }
}
function scan(list, initial, fun) {
    return scan_loop(list, initial, (0, _gleamMjs.toList)([]), fun);
}
function last(loop$list) {
    while(true){
        let list = loop$list;
        if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
        else {
            let $ = list.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let last$1 = list.head;
                return new (0, _gleamMjs.Ok)(last$1);
            } else {
                let rest$1 = $;
                loop$list = rest$1;
            }
        }
    }
}
function combinations(items, n) {
    if (n === 0) return (0, _gleamMjs.toList)([
        (0, _gleamMjs.toList)([])
    ]);
    else if (items instanceof (0, _gleamMjs.Empty)) return items;
    else {
        let first$1 = items.head;
        let rest$1 = items.tail;
        let _pipe = rest$1;
        let _pipe$1 = combinations(_pipe, n - 1);
        let _pipe$2 = map(_pipe$1, (combination)=>{
            return (0, _gleamMjs.prepend)(first$1, combination);
        });
        let _pipe$3 = reverse(_pipe$2);
        return fold(_pipe$3, combinations(rest$1, n), (acc, c)=>{
            return (0, _gleamMjs.prepend)(c, acc);
        });
    }
}
function combination_pairs_loop(loop$items, loop$acc) {
    while(true){
        let items = loop$items;
        let acc = loop$acc;
        if (items instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let first$1 = items.head;
            let rest$1 = items.tail;
            let first_combinations = map(rest$1, (other)=>{
                return [
                    first$1,
                    other
                ];
            });
            let acc$1 = reverse_and_prepend(first_combinations, acc);
            loop$items = rest$1;
            loop$acc = acc$1;
        }
    }
}
function combination_pairs(items) {
    return combination_pairs_loop(items, (0, _gleamMjs.toList)([]));
}
function take_firsts(loop$rows, loop$column, loop$remaining_rows) {
    while(true){
        let rows = loop$rows;
        let column = loop$column;
        let remaining_rows = loop$remaining_rows;
        if (rows instanceof (0, _gleamMjs.Empty)) return [
            reverse(column),
            reverse(remaining_rows)
        ];
        else {
            let $ = rows.head;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let rest$1 = rows.tail;
                loop$rows = rest$1;
                loop$column = column;
                loop$remaining_rows = remaining_rows;
            } else {
                let rest_rows = rows.tail;
                let first$1 = $.head;
                let remaining_row = $.tail;
                let remaining_rows$1 = (0, _gleamMjs.prepend)(remaining_row, remaining_rows);
                loop$rows = rest_rows;
                loop$column = (0, _gleamMjs.prepend)(first$1, column);
                loop$remaining_rows = remaining_rows$1;
            }
        }
    }
}
function transpose_loop(loop$rows, loop$columns) {
    while(true){
        let rows = loop$rows;
        let columns = loop$columns;
        if (rows instanceof (0, _gleamMjs.Empty)) return reverse(columns);
        else {
            let $ = take_firsts(rows, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
            let column;
            let rest$1;
            column = $[0];
            rest$1 = $[1];
            if (column instanceof (0, _gleamMjs.Empty)) {
                loop$rows = rest$1;
                loop$columns = columns;
            } else {
                loop$rows = rest$1;
                loop$columns = (0, _gleamMjs.prepend)(column, columns);
            }
        }
    }
}
function transpose(list_of_lists) {
    return transpose_loop(list_of_lists, (0, _gleamMjs.toList)([]));
}
function interleave(list) {
    let _pipe = list;
    let _pipe$1 = transpose(_pipe);
    return flatten(_pipe$1);
}
function shuffle_pair_unwrap_loop(loop$list, loop$acc) {
    while(true){
        let list = loop$list;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return acc;
        else {
            let elem_pair = list.head;
            let enumerable = list.tail;
            loop$list = enumerable;
            loop$acc = (0, _gleamMjs.prepend)(elem_pair[1], acc);
        }
    }
}
function do_shuffle_by_pair_indexes(list_of_pairs) {
    return sort(list_of_pairs, (a_pair, b_pair)=>{
        return _floatMjs.compare(a_pair[0], b_pair[0]);
    });
}
function shuffle(list) {
    let _pipe = list;
    let _pipe$1 = fold(_pipe, (0, _gleamMjs.toList)([]), (acc, a)=>{
        return (0, _gleamMjs.prepend)([
            _floatMjs.random(),
            a
        ], acc);
    });
    let _pipe$2 = do_shuffle_by_pair_indexes(_pipe$1);
    return shuffle_pair_unwrap_loop(_pipe$2, (0, _gleamMjs.toList)([]));
}
function max_loop(loop$list, loop$compare, loop$max) {
    while(true){
        let list = loop$list;
        let compare = loop$compare;
        let max = loop$max;
        if (list instanceof (0, _gleamMjs.Empty)) return max;
        else {
            let first$1 = list.head;
            let rest$1 = list.tail;
            let $ = compare(first$1, max);
            if ($ instanceof _orderMjs.Lt) {
                loop$list = rest$1;
                loop$compare = compare;
                loop$max = max;
            } else if ($ instanceof _orderMjs.Eq) {
                loop$list = rest$1;
                loop$compare = compare;
                loop$max = max;
            } else {
                loop$list = rest$1;
                loop$compare = compare;
                loop$max = first$1;
            }
        }
    }
}
function max(list, compare) {
    if (list instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Error)(undefined);
    else {
        let first$1 = list.head;
        let rest$1 = list.tail;
        return new (0, _gleamMjs.Ok)(max_loop(rest$1, compare, first$1));
    }
}
function build_reservoir_loop(loop$list, loop$size, loop$reservoir) {
    while(true){
        let list = loop$list;
        let size = loop$size;
        let reservoir = loop$reservoir;
        let reservoir_size = _dictMjs.size(reservoir);
        let $ = reservoir_size >= size;
        if ($) return [
            reservoir,
            list
        ];
        else {
            if (list instanceof (0, _gleamMjs.Empty)) return [
                reservoir,
                (0, _gleamMjs.toList)([])
            ];
            else {
                let first$1 = list.head;
                let rest$1 = list.tail;
                let reservoir$1 = _dictMjs.insert(reservoir, reservoir_size, first$1);
                loop$list = rest$1;
                loop$size = size;
                loop$reservoir = reservoir$1;
            }
        }
    }
}
/**
 * Builds the initial reservoir used by Algorithm L.
 * This is a dictionary with keys ranging from `0` up to `n - 1` where each
 * value is the corresponding element at that position in `list`.
 *
 * This also returns the remaining elements of `list` that didn't end up in
 * the reservoir.
 * 
 * @ignore
 */ function build_reservoir(list, n) {
    return build_reservoir_loop(list, n, _dictMjs.new$());
}
function log_random() {
    let $ = _floatMjs.logarithm(_floatMjs.random() + min_positive);
    let random;
    if ($ instanceof (0, _gleamMjs.Ok)) random = $[0];
    else throw (0, _gleamMjs.makeError)("let_assert", FILEPATH, "gleam/list", 2257, "log_random", "Pattern match failed, no pattern matched the value.", {
        value: $,
        start: 55515,
        end: 55586,
        pattern_start: 55526,
        pattern_end: 55536
    });
    return random;
}
function sample_loop(loop$list, loop$reservoir, loop$n, loop$w) {
    while(true){
        let list = loop$list;
        let reservoir = loop$reservoir;
        let n = loop$n;
        let w = loop$w;
        let _block;
        {
            let $ = _floatMjs.logarithm(1.0 - w);
            let log;
            if ($ instanceof (0, _gleamMjs.Ok)) log = $[0];
            else throw (0, _gleamMjs.makeError)("let_assert", FILEPATH, "gleam/list", 2240, "sample_loop", "Pattern match failed, no pattern matched the value.", {
                value: $,
                start: 55076,
                end: 55122,
                pattern_start: 55087,
                pattern_end: 55094
            });
            _block = _floatMjs.round(_floatMjs.floor((0, _gleamMjs.divideFloat)(log_random(), log)));
        }
        let skip = _block;
        let $ = drop(list, skip);
        if ($ instanceof (0, _gleamMjs.Empty)) return reservoir;
        else {
            let first$1 = $.head;
            let rest$1 = $.tail;
            let reservoir$1 = _dictMjs.insert(reservoir, _intMjs.random(n), first$1);
            let w$1 = w * _floatMjs.exponential((0, _gleamMjs.divideFloat)(log_random(), _intMjs.to_float(n)));
            loop$list = rest$1;
            loop$reservoir = reservoir$1;
            loop$n = n;
            loop$w = w$1;
        }
    }
}
function sample(list, n) {
    let $ = build_reservoir(list, n);
    let reservoir;
    let rest$1;
    reservoir = $[0];
    rest$1 = $[1];
    let $1 = _dictMjs.is_empty(reservoir);
    if ($1) return (0, _gleamMjs.toList)([]);
    else {
        let w = _floatMjs.exponential((0, _gleamMjs.divideFloat)(log_random(), _intMjs.to_float(n)));
        return _dictMjs.values(sample_loop(rest$1, reservoir, n, w));
    }
}
function permutation_zip(list, rest, acc) {
    if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
    else {
        let head = list.head;
        let tail = list.tail;
        return permutation_prepend(head, permutations(reverse_and_prepend(rest, tail)), tail, (0, _gleamMjs.prepend)(head, rest), acc);
    }
}
function permutations(list) {
    if (list instanceof (0, _gleamMjs.Empty)) return (0, _gleamMjs.toList)([
        (0, _gleamMjs.toList)([])
    ]);
    else {
        let l = list;
        return permutation_zip(l, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
    }
}
function permutation_prepend(loop$el, loop$permutations, loop$list_1, loop$list_2, loop$acc) {
    while(true){
        let el = loop$el;
        let permutations = loop$permutations;
        let list_1 = loop$list_1;
        let list_2 = loop$list_2;
        let acc = loop$acc;
        if (permutations instanceof (0, _gleamMjs.Empty)) return permutation_zip(list_1, list_2, acc);
        else {
            let head = permutations.head;
            let tail = permutations.tail;
            loop$el = el;
            loop$permutations = tail;
            loop$list_1 = list_1;
            loop$list_2 = list_2;
            loop$acc = (0, _gleamMjs.prepend)((0, _gleamMjs.prepend)(el, head), acc);
        }
    }
}

},{"../gleam.mjs":"aiPrb","../gleam/dict.mjs":"b8yrU","../gleam/float.mjs":"9bPI9","../gleam/int.mjs":"32hLf","../gleam/order.mjs":"eYj92","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"aiPrb":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"ib0cp":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "CustomType", ()=>CustomType);
parcelHelpers.export(exports, "List", ()=>List);
parcelHelpers.export(exports, "prepend", ()=>prepend);
parcelHelpers.export(exports, "toList", ()=>toList);
parcelHelpers.export(exports, "Empty", ()=>Empty);
parcelHelpers.export(exports, "List$Empty", ()=>List$Empty);
parcelHelpers.export(exports, "List$isEmpty", ()=>List$isEmpty);
parcelHelpers.export(exports, "NonEmpty", ()=>NonEmpty);
parcelHelpers.export(exports, "List$NonEmpty", ()=>List$NonEmpty);
parcelHelpers.export(exports, "List$isNonEmpty", ()=>List$isNonEmpty);
parcelHelpers.export(exports, "List$NonEmpty$first", ()=>List$NonEmpty$first);
parcelHelpers.export(exports, "List$NonEmpty$rest", ()=>List$NonEmpty$rest);
/**
 * A bit array is a contiguous sequence of bits similar to Erlang's Binary type.
 */ parcelHelpers.export(exports, "BitArray", ()=>BitArray);
parcelHelpers.export(exports, "BitArray$BitArray", ()=>BitArray$BitArray);
parcelHelpers.export(exports, "UtfCodepoint", ()=>UtfCodepoint);
/**
 * Slices a bit array to produce a new bit array. If `end` is not supplied then
 * all bits from `start` onward are returned.
 *
 * If the slice is out of bounds then an exception is thrown.
 *
 * @param {BitArray} bitArray
 * @param {number} start
 * @param {number} [end]
 * @returns {BitArray}
 */ parcelHelpers.export(exports, "bitArraySlice", ()=>bitArraySlice);
/**
 * Interprets a slice of this bit array as a floating point number, either
 * 32-bit or 64-bit, with the specified endianness.
 *
 * The value of `end - start` must be exactly 32 or 64, otherwise an exception
 * will be thrown.
 *
 * @param {BitArray} bitArray
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @returns {number}
 */ parcelHelpers.export(exports, "bitArraySliceToFloat", ()=>bitArraySliceToFloat);
/**
 * Interprets a slice of this bit array as a signed or unsigned integer with the
 * specified endianness.
 *
 * @param {BitArray} bitArray
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ parcelHelpers.export(exports, "bitArraySliceToInt", ()=>bitArraySliceToInt);
/**
 * Joins the given segments into a new bit array, tightly packing them together.
 * Each segment must be one of the following types:
 *
 * - A `number`: A single byte value in the range 0-255. Values outside this
 *   range will be wrapped.
 * - A `Uint8Array`: A sequence of byte values of any length.
 * - A `BitArray`: A sequence of bits of any length, which may not be byte
 *   aligned.
 *
 * The bit size of the returned bit array will be the sum of the size in bits
 * of the input segments.
 *
 * @param {(number | Uint8Array | BitArray)[]} segments
 * @returns {BitArray}
 */ parcelHelpers.export(exports, "toBitArray", ()=>toBitArray);
/**
 * Encodes a floating point value into a `Uint8Array`. This is used to create
 * float segments that are part of bit array expressions.
 *
 * @param {number} value
 * @param {number} size
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "sizedFloat", ()=>sizedFloat);
/**
 * Encodes an integer value into a `Uint8Array`, or a `BitArray` if the size in
 * bits is not a multiple of 8. This is used to create integer segments used in
 * bit array expressions.
 *
 * @param {number} value
 * @param {number} size
 * @param {boolean} isBigEndian
 * @returns {Uint8Array | BitArray}
 */ parcelHelpers.export(exports, "sizedInt", ()=>sizedInt);
/**
 * Returns the UTF-8 bytes for a string.
 *
 * @param {string} string
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "stringBits", ()=>stringBits);
/**
 * Returns the UTF-8 bytes for a single UTF codepoint.
 *
 * @param {UtfCodepoint} codepoint
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "codepointBits", ()=>codepointBits);
/**
 * Returns the UTF-16 bytes for a string.
 *
 * @param {string} string
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "stringToUtf16", ()=>stringToUtf16);
/**
 * Returns the UTF-16 bytes for a single UTF codepoint.
 *
 * @param {UtfCodepoint} codepoint
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "codepointToUtf16", ()=>codepointToUtf16);
/**
 * Returns the UTF-32 bytes for a string.
 *
 * @param {string} string
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "stringToUtf32", ()=>stringToUtf32);
/**
 * Returns the UTF-32 bytes for a single UTF codepoint.
 *
 * @param {UtfCodepoint} codepoint
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ parcelHelpers.export(exports, "codepointToUtf32", ()=>codepointToUtf32);
parcelHelpers.export(exports, "Result", ()=>Result);
parcelHelpers.export(exports, "Ok", ()=>Ok);
parcelHelpers.export(exports, "Result$Ok", ()=>Result$Ok);
parcelHelpers.export(exports, "Result$isOk", ()=>Result$isOk);
parcelHelpers.export(exports, "Result$Ok$0", ()=>Result$Ok$0);
parcelHelpers.export(exports, "Error", ()=>Error);
parcelHelpers.export(exports, "Result$Error", ()=>Result$Error);
parcelHelpers.export(exports, "Result$isError", ()=>Result$isError);
parcelHelpers.export(exports, "Result$Error$0", ()=>Result$Error$0);
parcelHelpers.export(exports, "isEqual", ()=>isEqual);
parcelHelpers.export(exports, "remainderInt", ()=>remainderInt);
parcelHelpers.export(exports, "divideInt", ()=>divideInt);
parcelHelpers.export(exports, "divideFloat", ()=>divideFloat);
parcelHelpers.export(exports, "makeError", ()=>makeError);
class CustomType {
    withFields(fields) {
        let properties = Object.keys(this).map((label)=>label in fields ? fields[label] : this[label]);
        return new this.constructor(...properties);
    }
}
class List {
    static fromArray(array, tail) {
        let t = tail || new Empty();
        for(let i = array.length - 1; i >= 0; --i)t = new NonEmpty(array[i], t);
        return t;
    }
    [Symbol.iterator]() {
        return new ListIterator(this);
    }
    toArray() {
        return [
            ...this
        ];
    }
    atLeastLength(desired) {
        let current = this;
        while(desired-- > 0 && current)current = current.tail;
        return current !== undefined;
    }
    hasLength(desired) {
        let current = this;
        while(desired-- > 0 && current)current = current.tail;
        return desired === -1 && current instanceof Empty;
    }
    countLength() {
        let current = this;
        let length = 0;
        while(current){
            current = current.tail;
            length++;
        }
        return length - 1;
    }
}
function prepend(element, tail) {
    return new NonEmpty(element, tail);
}
function toList(elements, tail) {
    return List.fromArray(elements, tail);
}
class ListIterator {
    #current;
    constructor(current){
        this.#current = current;
    }
    next() {
        if (this.#current instanceof Empty) return {
            done: true
        };
        else {
            let { head, tail } = this.#current;
            this.#current = tail;
            return {
                value: head,
                done: false
            };
        }
    }
}
class Empty extends List {
}
const List$Empty = ()=>new Empty();
const List$isEmpty = (value)=>value instanceof Empty;
class NonEmpty extends List {
    constructor(head, tail){
        super();
        this.head = head;
        this.tail = tail;
    }
}
const List$NonEmpty = (head, tail)=>new NonEmpty(head, tail);
const List$isNonEmpty = (value)=>value instanceof NonEmpty;
const List$NonEmpty$first = (value)=>value.head;
const List$NonEmpty$rest = (value)=>value.tail;
class BitArray {
    /**
   * The size in bits of this bit array's data.
   *
   * @type {number}
   */ bitSize;
    /**
   * The size in bytes of this bit array's data. If this bit array doesn't store
   * a whole number of bytes then this value is rounded up.
   *
   * @type {number}
   */ byteSize;
    /**
   * The number of unused high bits in the first byte of this bit array's
   * buffer prior to the start of its data. The value of any unused high bits is
   * undefined.
   *
   * The bit offset will be in the range 0-7.
   *
   * @type {number}
   */ bitOffset;
    /**
   * The raw bytes that hold this bit array's data.
   *
   * If `bitOffset` is not zero then there are unused high bits in the first
   * byte of this buffer.
   *
   * If `bitOffset + bitSize` is not a multiple of 8 then there are unused low
   * bits in the last byte of this buffer.
   *
   * @type {Uint8Array}
   */ rawBuffer;
    /**
   * Constructs a new bit array from a `Uint8Array`, an optional size in
   * bits, and an optional bit offset.
   *
   * If no bit size is specified it is taken as `buffer.length * 8`, i.e. all
   * bytes in the buffer make up the new bit array's data.
   *
   * If no bit offset is specified it defaults to zero, i.e. there are no unused
   * high bits in the first byte of the buffer.
   *
   * @param {Uint8Array} buffer
   * @param {number} [bitSize]
   * @param {number} [bitOffset]
   */ constructor(buffer, bitSize, bitOffset){
        if (!(buffer instanceof Uint8Array)) throw globalThis.Error("BitArray can only be constructed from a Uint8Array");
        this.bitSize = bitSize ?? buffer.length * 8;
        this.byteSize = Math.trunc((this.bitSize + 7) / 8);
        this.bitOffset = bitOffset ?? 0;
        // Validate the bit size
        if (this.bitSize < 0) throw globalThis.Error(`BitArray bit size is invalid: ${this.bitSize}`);
        // Validate the bit offset
        if (this.bitOffset < 0 || this.bitOffset > 7) throw globalThis.Error(`BitArray bit offset is invalid: ${this.bitOffset}`);
        // Validate the length of the buffer
        if (buffer.length !== Math.trunc((this.bitOffset + this.bitSize + 7) / 8)) throw globalThis.Error("BitArray buffer length is invalid");
        this.rawBuffer = buffer;
    }
    /**
   * Returns a specific byte in this bit array. If the byte index is out of
   * range then `undefined` is returned.
   *
   * When returning the final byte of a bit array with a bit size that's not a
   * multiple of 8, the content of the unused low bits are undefined.
   *
   * @param {number} index
   * @returns {number | undefined}
   */ byteAt(index) {
        if (index < 0 || index >= this.byteSize) return undefined;
        return bitArrayByteAt(this.rawBuffer, this.bitOffset, index);
    }
    equals(other) {
        if (this.bitSize !== other.bitSize) return false;
        const wholeByteCount = Math.trunc(this.bitSize / 8);
        // If both bit offsets are zero do a byte-aligned equality check which is
        // faster
        if (this.bitOffset === 0 && other.bitOffset === 0) {
            // Compare any whole bytes
            for(let i = 0; i < wholeByteCount; i++){
                if (this.rawBuffer[i] !== other.rawBuffer[i]) return false;
            }
            // Compare any trailing bits, excluding unused low bits
            const trailingBitsCount = this.bitSize % 8;
            if (trailingBitsCount) {
                const unusedLowBitCount = 8 - trailingBitsCount;
                if (this.rawBuffer[wholeByteCount] >> unusedLowBitCount !== other.rawBuffer[wholeByteCount] >> unusedLowBitCount) return false;
            }
        } else {
            // Compare any whole bytes
            for(let i = 0; i < wholeByteCount; i++){
                const a = bitArrayByteAt(this.rawBuffer, this.bitOffset, i);
                const b = bitArrayByteAt(other.rawBuffer, other.bitOffset, i);
                if (a !== b) return false;
            }
            // Compare any trailing bits
            const trailingBitsCount = this.bitSize % 8;
            if (trailingBitsCount) {
                const a = bitArrayByteAt(this.rawBuffer, this.bitOffset, wholeByteCount);
                const b = bitArrayByteAt(other.rawBuffer, other.bitOffset, wholeByteCount);
                const unusedLowBitCount = 8 - trailingBitsCount;
                if (a >> unusedLowBitCount !== b >> unusedLowBitCount) return false;
            }
        }
        return true;
    }
    /**
   * Returns this bit array's internal buffer.
   *
   * @deprecated Use `BitArray.byteAt()` or `BitArray.rawBuffer` instead.
   *
   * @returns {Uint8Array}
   */ get buffer() {
        bitArrayPrintDeprecationWarning("buffer", "Use BitArray.byteAt() or BitArray.rawBuffer instead");
        if (this.bitOffset !== 0 || this.bitSize % 8 !== 0) throw new globalThis.Error("BitArray.buffer does not support unaligned bit arrays");
        return this.rawBuffer;
    }
    /**
   * Returns the length in bytes of this bit array's internal buffer.
   *
   * @deprecated Use `BitArray.bitSize` or `BitArray.byteSize` instead.
   *
   * @returns {number}
   */ get length() {
        bitArrayPrintDeprecationWarning("length", "Use BitArray.bitSize or BitArray.byteSize instead");
        if (this.bitOffset !== 0 || this.bitSize % 8 !== 0) throw new globalThis.Error("BitArray.length does not support unaligned bit arrays");
        return this.rawBuffer.length;
    }
}
const BitArray$BitArray = (buffer, bitSize, bitOffset)=>new BitArray(buffer, bitSize, bitOffset);
/**
 * Returns the nth byte in the given buffer, after applying the specified bit
 * offset. If the index is out of bounds then zero is returned.
 *
 * @param {Uint8Array} buffer
 * @param {number} bitOffset
 * @param {number} index
 * @returns {number}
 */ function bitArrayByteAt(buffer, bitOffset, index) {
    if (bitOffset === 0) return buffer[index] ?? 0;
    else {
        const a = buffer[index] << bitOffset & 0xff;
        const b = buffer[index + 1] >> 8 - bitOffset;
        return a | b;
    }
}
class UtfCodepoint {
    constructor(value){
        this.value = value;
    }
}
const isBitArrayDeprecationMessagePrinted = {};
function bitArrayPrintDeprecationWarning(name, message) {
    if (isBitArrayDeprecationMessagePrinted[name]) return;
    console.warn(`Deprecated BitArray.${name} property used in JavaScript FFI code. ${message}.`);
    isBitArrayDeprecationMessagePrinted[name] = true;
}
function bitArraySlice(bitArray, start, end) {
    end ??= bitArray.bitSize;
    bitArrayValidateRange(bitArray, start, end);
    // Handle zero-length slices
    if (start === end) return new BitArray(new Uint8Array());
    // Early return for slices that cover the whole bit array
    if (start === 0 && end === bitArray.bitSize) return bitArray;
    start += bitArray.bitOffset;
    end += bitArray.bitOffset;
    const startByteIndex = Math.trunc(start / 8);
    const endByteIndex = Math.trunc((end + 7) / 8);
    const byteLength = endByteIndex - startByteIndex;
    // Avoid creating a new Uint8Array if the view of the underlying ArrayBuffer
    // is the same. This can occur when slicing off just the first or last bit of
    // a bit array, i.e. when only the bit offset or bit size need to be updated.
    let buffer;
    if (startByteIndex === 0 && byteLength === bitArray.rawBuffer.byteLength) buffer = bitArray.rawBuffer;
    else buffer = new Uint8Array(bitArray.rawBuffer.buffer, bitArray.rawBuffer.byteOffset + startByteIndex, byteLength);
    return new BitArray(buffer, end - start, start % 8);
}
function bitArraySliceToFloat(bitArray, start, end, isBigEndian) {
    bitArrayValidateRange(bitArray, start, end);
    const floatSize = end - start;
    // Check size is valid
    if (floatSize !== 16 && floatSize !== 32 && floatSize !== 64) {
        const msg = `Sized floats must be 16-bit, 32-bit or 64-bit, got size of ` + `${floatSize} bits`;
        throw new globalThis.Error(msg);
    }
    start += bitArray.bitOffset;
    const isStartByteAligned = start % 8 === 0;
    // If the bit range is byte aligned then the float can be read directly out
    // of the existing buffer
    if (isStartByteAligned) {
        const view = new DataView(bitArray.rawBuffer.buffer, bitArray.rawBuffer.byteOffset + start / 8);
        if (floatSize === 64) return view.getFloat64(0, !isBigEndian);
        else if (floatSize === 32) return view.getFloat32(0, !isBigEndian);
        else if (floatSize === 16) return fp16UintToNumber(view.getUint16(0, !isBigEndian));
    }
    // Copy the unaligned bytes into an aligned array so a DataView can be used
    const alignedBytes = new Uint8Array(floatSize / 8);
    const byteOffset = Math.trunc(start / 8);
    for(let i = 0; i < alignedBytes.length; i++)alignedBytes[i] = bitArrayByteAt(bitArray.rawBuffer, start % 8, byteOffset + i);
    // Read the float out of the aligned buffer
    const view = new DataView(alignedBytes.buffer);
    if (floatSize === 64) return view.getFloat64(0, !isBigEndian);
    else if (floatSize === 32) return view.getFloat32(0, !isBigEndian);
    else return fp16UintToNumber(view.getUint16(0, !isBigEndian));
}
function bitArraySliceToInt(bitArray, start, end, isBigEndian, isSigned) {
    bitArrayValidateRange(bitArray, start, end);
    if (start === end) return 0;
    start += bitArray.bitOffset;
    end += bitArray.bitOffset;
    const isStartByteAligned = start % 8 === 0;
    const isEndByteAligned = end % 8 === 0;
    // If the slice is byte-aligned then there is no need to handle unaligned
    // slices, meaning a simpler and faster implementation can be used instead
    if (isStartByteAligned && isEndByteAligned) return intFromAlignedSlice(bitArray, start / 8, end / 8, isBigEndian, isSigned);
    const size = end - start;
    const startByteIndex = Math.trunc(start / 8);
    const endByteIndex = Math.trunc((end - 1) / 8);
    // Handle the case of the slice being completely contained in a single byte
    if (startByteIndex == endByteIndex) {
        const mask = 0xff >> start % 8;
        const unusedLowBitCount = (8 - end % 8) % 8;
        let value = (bitArray.rawBuffer[startByteIndex] & mask) >> unusedLowBitCount;
        // For signed integers, if the high bit is set reinterpret as two's
        // complement
        if (isSigned) {
            const highBit = 2 ** (size - 1);
            if (value >= highBit) value -= highBit * 2;
        }
        return value;
    }
    // The integer value to be read is not aligned and crosses at least one byte
    // boundary in the input array
    if (size <= 53) return intFromUnalignedSliceUsingNumber(bitArray.rawBuffer, start, end, isBigEndian, isSigned);
    else return intFromUnalignedSliceUsingBigInt(bitArray.rawBuffer, start, end, isBigEndian, isSigned);
}
function toBitArray(segments) {
    if (segments.length === 0) return new BitArray(new Uint8Array());
    if (segments.length === 1) {
        const segment = segments[0];
        // When there is a single BitArray segment it can be returned as-is
        if (segment instanceof BitArray) return segment;
        // When there is a single Uint8Array segment, pass it directly to the bit
        // array constructor to avoid a copy
        if (segment instanceof Uint8Array) return new BitArray(segment);
        return new BitArray(new Uint8Array(/** @type {number[]} */ segments));
    }
    // Count the total number of bits and check if all segments are numbers, i.e.
    // single bytes
    let bitSize = 0;
    let areAllSegmentsNumbers = true;
    for (const segment of segments){
        if (segment instanceof BitArray) {
            bitSize += segment.bitSize;
            areAllSegmentsNumbers = false;
        } else if (segment instanceof Uint8Array) {
            bitSize += segment.byteLength * 8;
            areAllSegmentsNumbers = false;
        } else bitSize += 8;
    }
    // If all segments are numbers then pass the segments array directly to the
    // Uint8Array constructor
    if (areAllSegmentsNumbers) return new BitArray(new Uint8Array(/** @type {number[]} */ segments));
    // Pack the segments into a Uint8Array
    const buffer = new Uint8Array(Math.trunc((bitSize + 7) / 8));
    // The current write position in bits into the above array. Byte-aligned
    // segments, i.e. when the cursor is a multiple of 8, are able to be processed
    // faster due to being able to copy bytes directly.
    let cursor = 0;
    for (let segment of segments){
        const isCursorByteAligned = cursor % 8 === 0;
        if (segment instanceof BitArray) {
            if (isCursorByteAligned && segment.bitOffset === 0) {
                buffer.set(segment.rawBuffer, cursor / 8);
                cursor += segment.bitSize;
                // Zero any unused bits in the last byte of the buffer. Their content is
                // undefined and shouldn't be included in the output.
                const trailingBitsCount = segment.bitSize % 8;
                if (trailingBitsCount !== 0) {
                    const lastByteIndex = Math.trunc(cursor / 8);
                    buffer[lastByteIndex] >>= 8 - trailingBitsCount;
                    buffer[lastByteIndex] <<= 8 - trailingBitsCount;
                }
            } else appendUnalignedBits(segment.rawBuffer, segment.bitSize, segment.bitOffset);
        } else if (segment instanceof Uint8Array) {
            if (isCursorByteAligned) {
                buffer.set(segment, cursor / 8);
                cursor += segment.byteLength * 8;
            } else appendUnalignedBits(segment, segment.byteLength * 8, 0);
        } else if (isCursorByteAligned) {
            buffer[cursor / 8] = segment;
            cursor += 8;
        } else appendUnalignedBits(new Uint8Array([
            segment
        ]), 8, 0);
    }
    function appendUnalignedBits(unalignedBits, size, offset) {
        if (size === 0) return;
        const byteSize = Math.trunc(size + 7 / 8);
        const highBitsCount = cursor % 8;
        const lowBitsCount = 8 - highBitsCount;
        let byteIndex = Math.trunc(cursor / 8);
        for(let i = 0; i < byteSize; i++){
            let byte = bitArrayByteAt(unalignedBits, offset, i);
            // If this is a partial byte then zero out the trailing bits as their
            // content is undefined and shouldn't be included in the output
            if (size < 8) {
                byte >>= 8 - size;
                byte <<= 8 - size;
            }
            // Copy the high bits of the input byte to the low bits of the current
            // output byte
            buffer[byteIndex] |= byte >> highBitsCount;
            let appendedBitsCount = size - Math.max(0, size - lowBitsCount);
            size -= appendedBitsCount;
            cursor += appendedBitsCount;
            if (size === 0) break;
            // Copy the low bits of the input byte to the high bits of the next output
            // byte
            buffer[++byteIndex] = byte << lowBitsCount;
            appendedBitsCount = size - Math.max(0, size - highBitsCount);
            size -= appendedBitsCount;
            cursor += appendedBitsCount;
        }
    }
    return new BitArray(buffer, bitSize);
}
function sizedFloat(value, size, isBigEndian) {
    if (size !== 16 && size !== 32 && size !== 64) {
        const msg = `Sized floats must be 16-bit, 32-bit or 64-bit, got size of ${size} bits`;
        throw new globalThis.Error(msg);
    }
    if (size === 16) return numberToFp16Uint(value, isBigEndian);
    const buffer = new Uint8Array(size / 8);
    const view = new DataView(buffer.buffer);
    if (size == 64) view.setFloat64(0, value, !isBigEndian);
    else view.setFloat32(0, value, !isBigEndian);
    return buffer;
}
function sizedInt(value, size, isBigEndian) {
    if (size <= 0) return new Uint8Array();
    // Fast path when size is 8 bits. This relies on the rounding behavior of the
    // Uint8Array constructor.
    if (size === 8) return new Uint8Array([
        value
    ]);
    // Fast path when size is less than 8 bits: shift the value up to the high
    // bits
    if (size < 8) {
        value <<= 8 - size;
        return new BitArray(new Uint8Array([
            value
        ]), size);
    }
    // Allocate output buffer
    const buffer = new Uint8Array(Math.trunc((size + 7) / 8));
    // The number of trailing bits in the final byte. Will be zero if the size is
    // an exact number of bytes.
    const trailingBitsCount = size % 8;
    // The number of unused bits in the final byte of the buffer
    const unusedBitsCount = 8 - trailingBitsCount;
    // For output sizes not exceeding 32 bits the number type is used. For larger
    // output sizes the BigInt type is needed.
    //
    // The code in each of these two paths must be kept in sync.
    if (size <= 32) {
        if (isBigEndian) {
            let i = buffer.length - 1;
            // Set the trailing bits at the end of the output buffer
            if (trailingBitsCount) {
                buffer[i--] = value << unusedBitsCount & 0xff;
                value >>= trailingBitsCount;
            }
            for(; i >= 0; i--){
                buffer[i] = value;
                value >>= 8;
            }
        } else {
            let i = 0;
            const wholeByteCount = Math.trunc(size / 8);
            for(; i < wholeByteCount; i++){
                buffer[i] = value;
                value >>= 8;
            }
            // Set the trailing bits at the end of the output buffer
            if (trailingBitsCount) buffer[i] = value << unusedBitsCount;
        }
    } else {
        const bigTrailingBitsCount = BigInt(trailingBitsCount);
        const bigUnusedBitsCount = BigInt(unusedBitsCount);
        let bigValue = BigInt(value);
        if (isBigEndian) {
            let i = buffer.length - 1;
            // Set the trailing bits at the end of the output buffer
            if (trailingBitsCount) {
                buffer[i--] = Number(bigValue << bigUnusedBitsCount);
                bigValue >>= bigTrailingBitsCount;
            }
            for(; i >= 0; i--){
                buffer[i] = Number(bigValue);
                bigValue >>= 8n;
            }
        } else {
            let i = 0;
            const wholeByteCount = Math.trunc(size / 8);
            for(; i < wholeByteCount; i++){
                buffer[i] = Number(bigValue);
                bigValue >>= 8n;
            }
            // Set the trailing bits at the end of the output buffer
            if (trailingBitsCount) buffer[i] = Number(bigValue << bigUnusedBitsCount);
        }
    }
    // Integers that aren't a whole number of bytes are returned as a BitArray so
    // their size in bits is tracked
    if (trailingBitsCount) return new BitArray(buffer, size);
    return buffer;
}
/**
 * Reads an aligned slice of any size as an integer.
 *
 * @param {BitArray} bitArray
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ function intFromAlignedSlice(bitArray, start, end, isBigEndian, isSigned) {
    const byteSize = end - start;
    if (byteSize <= 6) return intFromAlignedSliceUsingNumber(bitArray.rawBuffer, start, end, isBigEndian, isSigned);
    else return intFromAlignedSliceUsingBigInt(bitArray.rawBuffer, start, end, isBigEndian, isSigned);
}
/**
 * Reads an aligned slice up to 48 bits in size as an integer. Uses the
 * JavaScript `number` type internally.
 *
 * @param {Uint8Array} buffer
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ function intFromAlignedSliceUsingNumber(buffer, start, end, isBigEndian, isSigned) {
    const byteSize = end - start;
    let value = 0;
    // Read bytes as an unsigned integer
    if (isBigEndian) for(let i = start; i < end; i++){
        value *= 256;
        value += buffer[i];
    }
    else for(let i = end - 1; i >= start; i--){
        value *= 256;
        value += buffer[i];
    }
    // For signed integers, if the high bit is set reinterpret as two's
    // complement
    if (isSigned) {
        const highBit = 2 ** (byteSize * 8 - 1);
        if (value >= highBit) value -= highBit * 2;
    }
    return value;
}
/**
 * Reads an aligned slice of any size as an integer. Uses the JavaScript
 * `BigInt` type internally.
 *
 * @param {Uint8Array} buffer
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ function intFromAlignedSliceUsingBigInt(buffer, start, end, isBigEndian, isSigned) {
    const byteSize = end - start;
    let value = 0n;
    // Read bytes as an unsigned integer value
    if (isBigEndian) for(let i = start; i < end; i++){
        value *= 256n;
        value += BigInt(buffer[i]);
    }
    else for(let i = end - 1; i >= start; i--){
        value *= 256n;
        value += BigInt(buffer[i]);
    }
    // For signed integers, if the high bit is set reinterpret as two's
    // complement
    if (isSigned) {
        const highBit = 1n << BigInt(byteSize * 8 - 1);
        if (value >= highBit) value -= highBit * 2n;
    }
    // Convert the result into a JS number. This may cause quantizing/error on
    // values outside JavaScript's safe integer range.
    return Number(value);
}
/**
 * Reads an unaligned slice up to 53 bits in size as an integer. Uses the
 * JavaScript `number` type internally.
 *
 * This function assumes that the slice crosses at least one byte boundary in
 * the input.
 *
 * @param {Uint8Array} buffer
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ function intFromUnalignedSliceUsingNumber(buffer, start, end, isBigEndian, isSigned) {
    const isStartByteAligned = start % 8 === 0;
    let size = end - start;
    let byteIndex = Math.trunc(start / 8);
    let value = 0;
    if (isBigEndian) {
        // Read any leading bits
        if (!isStartByteAligned) {
            const leadingBitsCount = 8 - start % 8;
            value = buffer[byteIndex++] & (1 << leadingBitsCount) - 1;
            size -= leadingBitsCount;
        }
        // Read any whole bytes
        while(size >= 8){
            value *= 256;
            value += buffer[byteIndex++];
            size -= 8;
        }
        // Read any trailing bits
        if (size > 0) {
            value *= 2 ** size;
            value += buffer[byteIndex] >> 8 - size;
        }
    } else // For little endian, if the start is aligned then whole bytes can be read
    // directly out of the input array, with the trailing bits handled at the
    // end
    if (isStartByteAligned) {
        let size = end - start;
        let scale = 1;
        // Read whole bytes
        while(size >= 8){
            value += buffer[byteIndex++] * scale;
            scale *= 256;
            size -= 8;
        }
        // Read trailing bits
        value += (buffer[byteIndex] >> 8 - size) * scale;
    } else {
        // Read little endian data where the start is not byte-aligned. This is
        // done by reading whole bytes that cross a byte boundary in the input
        // data, then reading any trailing bits.
        const highBitsCount = start % 8;
        const lowBitsCount = 8 - highBitsCount;
        let size = end - start;
        let scale = 1;
        // Extract whole bytes
        while(size >= 8){
            const byte = buffer[byteIndex] << highBitsCount | buffer[byteIndex + 1] >> lowBitsCount;
            value += (byte & 0xff) * scale;
            scale *= 256;
            size -= 8;
            byteIndex++;
        }
        // Read any trailing bits. These trailing bits may cross a byte boundary
        // in the input buffer.
        if (size > 0) {
            const lowBitsUsed = size - Math.max(0, size - lowBitsCount);
            let trailingByte = (buffer[byteIndex] & (1 << lowBitsCount) - 1) >> lowBitsCount - lowBitsUsed;
            size -= lowBitsUsed;
            if (size > 0) {
                trailingByte *= 2 ** size;
                trailingByte += buffer[byteIndex + 1] >> 8 - size;
            }
            value += trailingByte * scale;
        }
    }
    // For signed integers, if the high bit is set reinterpret as two's
    // complement
    if (isSigned) {
        const highBit = 2 ** (end - start - 1);
        if (value >= highBit) value -= highBit * 2;
    }
    return value;
}
/**
 * Reads an unaligned slice of any size as an integer. Uses the JavaScript
 * `BigInt` type internally.
 *
 * This function assumes that the slice crosses at least one byte boundary in
 * the input.
 *
 * @param {Uint8Array} buffer
 * @param {number} start
 * @param {number} end
 * @param {boolean} isBigEndian
 * @param {boolean} isSigned
 * @returns {number}
 */ function intFromUnalignedSliceUsingBigInt(buffer, start, end, isBigEndian, isSigned) {
    const isStartByteAligned = start % 8 === 0;
    let size = end - start;
    let byteIndex = Math.trunc(start / 8);
    let value = 0n;
    if (isBigEndian) {
        // Read any leading bits
        if (!isStartByteAligned) {
            const leadingBitsCount = 8 - start % 8;
            value = BigInt(buffer[byteIndex++] & (1 << leadingBitsCount) - 1);
            size -= leadingBitsCount;
        }
        // Read any whole bytes
        while(size >= 8){
            value *= 256n;
            value += BigInt(buffer[byteIndex++]);
            size -= 8;
        }
        // Read any trailing bits
        if (size > 0) {
            value <<= BigInt(size);
            value += BigInt(buffer[byteIndex] >> 8 - size);
        }
    } else // For little endian, if the start is aligned then whole bytes can be read
    // directly out of the input array, with the trailing bits handled at the
    // end
    if (isStartByteAligned) {
        let size = end - start;
        let shift = 0n;
        // Read whole bytes
        while(size >= 8){
            value += BigInt(buffer[byteIndex++]) << shift;
            shift += 8n;
            size -= 8;
        }
        // Read trailing bits
        value += BigInt(buffer[byteIndex] >> 8 - size) << shift;
    } else {
        // Read little endian data where the start is not byte-aligned. This is
        // done by reading whole bytes that cross a byte boundary in the input
        // data, then reading any trailing bits.
        const highBitsCount = start % 8;
        const lowBitsCount = 8 - highBitsCount;
        let size = end - start;
        let shift = 0n;
        // Extract whole bytes
        while(size >= 8){
            const byte = buffer[byteIndex] << highBitsCount | buffer[byteIndex + 1] >> lowBitsCount;
            value += BigInt(byte & 0xff) << shift;
            shift += 8n;
            size -= 8;
            byteIndex++;
        }
        // Read any trailing bits. These trailing bits may cross a byte boundary
        // in the input buffer.
        if (size > 0) {
            const lowBitsUsed = size - Math.max(0, size - lowBitsCount);
            let trailingByte = (buffer[byteIndex] & (1 << lowBitsCount) - 1) >> lowBitsCount - lowBitsUsed;
            size -= lowBitsUsed;
            if (size > 0) {
                trailingByte <<= size;
                trailingByte += buffer[byteIndex + 1] >> 8 - size;
            }
            value += BigInt(trailingByte) << shift;
        }
    }
    // For signed integers, if the high bit is set reinterpret as two's
    // complement
    if (isSigned) {
        const highBit = 2n ** BigInt(end - start - 1);
        if (value >= highBit) value -= highBit * 2n;
    }
    // Convert the result into a JS number. This may cause quantizing/error on
    // values outside JavaScript's safe integer range.
    return Number(value);
}
/**
 * Interprets a 16-bit unsigned integer value as a 16-bit floating point value.
 *
 * @param {number} intValue
 * @returns {number}
 */ function fp16UintToNumber(intValue) {
    const sign = intValue >= 0x8000 ? -1 : 1;
    const exponent = (intValue & 0x7c00) >> 10;
    const fraction = intValue & 0x03ff;
    let value;
    if (exponent === 0) value = 6.103515625e-5 * (fraction / 0x400);
    else if (exponent === 0x1f) value = fraction === 0 ? Infinity : NaN;
    else value = Math.pow(2, exponent - 15) * (1 + fraction / 0x400);
    return sign * value;
}
/**
 * Converts a floating point number to bytes for a 16-bit floating point value.
 *
 * @param {number} intValue
 * @param {boolean} isBigEndian
 * @returns {Uint8Array}
 */ function numberToFp16Uint(value, isBigEndian) {
    const buffer = new Uint8Array(2);
    if (isNaN(value)) buffer[1] = 0x7e;
    else if (value === Infinity) buffer[1] = 0x7c;
    else if (value === -Infinity) buffer[1] = 0xfc;
    else if (value === 0) ;
    else {
        const sign = value < 0 ? 1 : 0;
        value = Math.abs(value);
        let exponent = Math.floor(Math.log2(value));
        let fraction = value / Math.pow(2, exponent) - 1;
        exponent += 15;
        if (exponent <= 0) {
            exponent = 0;
            fraction = value / Math.pow(2, -14);
        } else if (exponent >= 31) {
            exponent = 31;
            fraction = 0;
        }
        fraction = Math.round(fraction * 1024);
        buffer[1] = sign << 7 | (exponent & 0x1f) << 2 | fraction >> 8 & 0x03;
        buffer[0] = fraction & 0xff;
    }
    if (isBigEndian) {
        const a = buffer[0];
        buffer[0] = buffer[1];
        buffer[1] = a;
    }
    return buffer;
}
/**
 * Throws an exception if the given start and end values are out of bounds for
 * a bit array.
 *
 * @param {BitArray} bitArray
 * @param {number} start
 * @param {number} end
 */ function bitArrayValidateRange(bitArray, start, end) {
    if (start < 0 || start > bitArray.bitSize || end < start || end > bitArray.bitSize) {
        const msg = `Invalid bit array slice: start = ${start}, end = ${end}, ` + `bit size = ${bitArray.bitSize}`;
        throw new globalThis.Error(msg);
    }
}
/** @type {TextEncoder | undefined} */ let utf8Encoder;
function stringBits(string) {
    utf8Encoder ??= new TextEncoder();
    return utf8Encoder.encode(string);
}
function codepointBits(codepoint) {
    return stringBits(String.fromCodePoint(codepoint.value));
}
function stringToUtf16(string, isBigEndian) {
    const buffer = new ArrayBuffer(string.length * 2);
    const bufferView = new DataView(buffer);
    for(let i = 0; i < string.length; i++)bufferView.setUint16(i * 2, string.charCodeAt(i), !isBigEndian);
    return new Uint8Array(buffer);
}
function codepointToUtf16(codepoint, isBigEndian) {
    return stringToUtf16(String.fromCodePoint(codepoint.value), isBigEndian);
}
function stringToUtf32(string, isBigEndian) {
    const buffer = new ArrayBuffer(string.length * 4);
    const bufferView = new DataView(buffer);
    let length = 0;
    for(let i = 0; i < string.length; i++){
        const codepoint = string.codePointAt(i);
        bufferView.setUint32(length * 4, codepoint, !isBigEndian);
        length++;
        if (codepoint > 0xffff) i++;
    }
    return new Uint8Array(buffer.slice(0, length * 4));
}
function codepointToUtf32(codepoint, isBigEndian) {
    return stringToUtf32(String.fromCodePoint(codepoint.value), isBigEndian);
}
class Result extends CustomType {
    static isResult(data) {
        return data instanceof Result;
    }
}
class Ok extends Result {
    constructor(value){
        super();
        this[0] = value;
    }
    isOk() {
        return true;
    }
}
const Result$Ok = (value)=>new Ok(value);
const Result$isOk = (value)=>value instanceof Ok;
const Result$Ok$0 = (value)=>value[0];
class Error extends Result {
    constructor(detail){
        super();
        this[0] = detail;
    }
    isOk() {
        return false;
    }
}
const Result$Error = (detail)=>new Error(detail);
const Result$isError = (value)=>value instanceof Error;
const Result$Error$0 = (value)=>value[0];
function isEqual(x, y) {
    let values = [
        x,
        y
    ];
    while(values.length){
        let a = values.pop();
        let b = values.pop();
        if (a === b) continue;
        if (!isObject(a) || !isObject(b)) return false;
        let unequal = !structurallyCompatibleObjects(a, b) || unequalDates(a, b) || unequalBuffers(a, b) || unequalArrays(a, b) || unequalMaps(a, b) || unequalSets(a, b) || unequalRegExps(a, b);
        if (unequal) return false;
        const proto = Object.getPrototypeOf(a);
        if (proto !== null && typeof proto.equals === "function") try {
            if (a.equals(b)) continue;
            else return false;
        } catch  {}
        let [keys, get] = getters(a);
        const ka = keys(a);
        const kb = keys(b);
        if (ka.length !== kb.length) return false;
        for (let k of ka)values.push(get(a, k), get(b, k));
    }
    return true;
}
function getters(object) {
    if (object instanceof Map) return [
        (x)=>x.keys(),
        (x, y)=>x.get(y)
    ];
    else {
        let extra = object instanceof globalThis.Error ? [
            "message"
        ] : [];
        return [
            (x)=>[
                    ...extra,
                    ...Object.keys(x)
                ],
            (x, y)=>x[y]
        ];
    }
}
function unequalDates(a, b) {
    return a instanceof Date && (a > b || a < b);
}
function unequalBuffers(a, b) {
    return !(a instanceof BitArray) && a.buffer instanceof ArrayBuffer && a.BYTES_PER_ELEMENT && !(a.byteLength === b.byteLength && a.every((n, i)=>n === b[i]));
}
function unequalArrays(a, b) {
    return Array.isArray(a) && a.length !== b.length;
}
function unequalMaps(a, b) {
    return a instanceof Map && a.size !== b.size;
}
function unequalSets(a, b) {
    return a instanceof Set && (a.size != b.size || [
        ...a
    ].some((e)=>!b.has(e)));
}
function unequalRegExps(a, b) {
    return a instanceof RegExp && (a.source !== b.source || a.flags !== b.flags);
}
function isObject(a) {
    return typeof a === "object" && a !== null;
}
function structurallyCompatibleObjects(a, b) {
    if (typeof a !== "object" && typeof b !== "object" && (!a || !b)) return false;
    let nonstructural = [
        Promise,
        WeakSet,
        WeakMap,
        Function
    ];
    if (nonstructural.some((c)=>a instanceof c)) return false;
    return a.constructor === b.constructor;
}
function remainderInt(a, b) {
    if (b === 0) return 0;
    else return a % b;
}
function divideInt(a, b) {
    return Math.trunc(divideFloat(a, b));
}
function divideFloat(a, b) {
    if (b === 0) return 0;
    else return a / b;
}
function makeError(variant, file, module, line, fn, message, extra) {
    let error = new globalThis.Error(message);
    error.gleam_error = variant;
    error.file = file;
    error.module = module;
    error.line = line;
    error.function = fn;
    // TODO: Remove this with Gleam v2.0.0
    error.fn = fn;
    for(let k in extra)error[k] = extra[k];
    return error;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jnFvT":[function(require,module,exports,__globalThis) {
exports.interopDefault = function(a) {
    return a && a.__esModule ? a : {
        default: a
    };
};
exports.defineInteropFlag = function(a) {
    Object.defineProperty(a, '__esModule', {
        value: true
    });
};
exports.exportAll = function(source, dest) {
    Object.keys(source).forEach(function(key) {
        if (key === 'default' || key === '__esModule' || Object.prototype.hasOwnProperty.call(dest, key)) return;
        Object.defineProperty(dest, key, {
            enumerable: true,
            get: function() {
                return source[key];
            }
        });
    });
    return dest;
};
exports.export = function(dest, destName, get) {
    Object.defineProperty(dest, destName, {
        enumerable: true,
        get: get
    });
};

},{}],"b8yrU":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "fold", ()=>(0, _dictMjs.fold));
parcelHelpers.export(exports, "get", ()=>(0, _dictMjs.get));
parcelHelpers.export(exports, "has_key", ()=>(0, _dictMjs.has));
parcelHelpers.export(exports, "insert", ()=>(0, _dictMjs.insert));
parcelHelpers.export(exports, "map_values", ()=>(0, _dictMjs.map));
parcelHelpers.export(exports, "new$", ()=>(0, _dictMjs.make));
parcelHelpers.export(exports, "size", ()=>(0, _dictMjs.size));
/**
 * Determines whether or not the dict is empty.
 *
 * ## Examples
 *
 * ```gleam
 * assert new() |> is_empty
 * ```
 *
 * ```gleam
 * assert !{ new() |> insert("b", 1) |> is_empty }
 * ```
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
/**
 * Converts a list of 2-element tuples `#(key, value)` to a dict.
 *
 * If two tuples have the same key the last one in the list will be the one
 * that is present in the dict.
 */ parcelHelpers.export(exports, "from_list", ()=>from_list);
/**
 * Creates a new dict from a given dict, only including any entries for which the
 * keys are in a given list.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)])
 *   |> take(["b"])
 *   == from_list([#("b", 1)])
 * ```
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)])
 *   |> take(["a", "b", "c"])
 *   == from_list([#("a", 0), #("b", 1)])
 * ```
 */ parcelHelpers.export(exports, "take", ()=>take);
/**
 * Creates a new dict from a given dict with all the same entries except for the
 * one with a given key, if it exists.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> delete("a")
 *   == from_list([#("b", 1)])
 * ```
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> delete("c")
 *   == from_list([#("a", 0), #("b", 1)])
 * ```
 */ parcelHelpers.export(exports, "delete$", ()=>delete$);
/**
 * Creates a new dict from a given dict with all the same entries except any with
 * keys found in a given list.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> drop(["a"])
 *   == from_list([#("b", 1)])
 * ```
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> drop(["c"])
 *   == from_list([#("a", 0), #("b", 1)])
 * ```
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> drop(["a", "b", "c"])
 *   == from_list([])
 * ```
 */ parcelHelpers.export(exports, "drop", ()=>drop);
/**
 * Creates a new dict with one entry inserted or updated using a given function.
 *
 * If there was not an entry in the dict for the given key then the function
 * gets `None` as its argument, otherwise it gets `Some(value)`.
 *
 * ## Examples
 *
 * ```gleam
 * let dict = from_list([#("a", 0)])
 * let increment = fn(x) {
 *   case x {
 *     Some(i) -> i + 1
 *     None -> 0
 *   }
 * }
 *
 * assert upsert(dict, "a", increment) == from_list([#("a", 1)])
 * ```
 *
 * ```gleam
 * assert upsert(dict, "b", increment) == from_list([#("a", 0), #("b", 0)])
 * ```
 */ parcelHelpers.export(exports, "upsert", ()=>upsert);
/**
 * Converts the dict to a list of 2-element tuples `#(key, value)`, one for
 * each key-value pair in the dict.
 *
 * The tuples in the list have no specific order.
 *
 * ## Examples
 *
 * Calling `to_list` on an empty `dict` returns an empty list.
 *
 * ```gleam
 * assert new() |> to_list == []
 * ```
 *
 * The ordering of elements in the resulting list is an implementation detail
 * that should not be relied upon.
 *
 * ```gleam
 * assert new()
 *   |> insert("b", 1)
 *   |> insert("a", 0)
 *   |> insert("c", 2)
 *   |> to_list
 *   == [#("a", 0), #("b", 1), #("c", 2)]
 * ```
 */ parcelHelpers.export(exports, "to_list", ()=>to_list);
/**
 * Gets a list of all keys in a given dict.
 *
 * Dicts are not ordered so the keys are not returned in any specific order. Do
 * not write code that relies on the order keys are returned by this function
 * as it may change in later versions of Gleam or Erlang.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> keys == ["a", "b"]
 * ```
 */ parcelHelpers.export(exports, "keys", ()=>keys);
/**
 * Gets a list of all values in a given dict.
 *
 * Dicts are not ordered so the values are not returned in any specific order. Do
 * not write code that relies on the order values are returned by this function
 * as it may change in later versions of Gleam or Erlang.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)]) |> values == [0, 1]
 * ```
 */ parcelHelpers.export(exports, "values", ()=>values);
/**
 * Creates a new dict from a given dict, minus any entries that a given function
 * returns `False` for.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)])
 *   |> filter(fn(key, value) { value != 0 })
 *   == from_list([#("b", 1)])
 * ```
 *
 * ```gleam
 * assert from_list([#("a", 0), #("b", 1)])
 *   |> filter(fn(key, value) { True })
 *   == from_list([#("a", 0), #("b", 1)])
 * ```
 */ parcelHelpers.export(exports, "filter", ()=>filter);
/**
 * Calls a function for each key and value in a dict, discarding the return
 * value.
 *
 * Useful for producing a side effect for every item of a dict.
 *
 * ```gleam
 * import gleam/io
 *
 * let dict = from_list([#("a", "apple"), #("b", "banana"), #("c", "cherry")])
 *
 * assert
 *   each(dict, fn(k, v) {
 *     io.println(k <> " => " <> v)
 *   })
 *   == Nil
 * // a => apple
 * // b => banana
 * // c => cherry
 * ```
 *
 * The order of elements in the iteration is an implementation detail that
 * should not be relied upon.
 */ parcelHelpers.export(exports, "each", ()=>each);
/**
 * Creates a new dict from a pair of given dicts by combining their entries.
 *
 * If there are entries with the same keys in both dicts the given function is
 * used to determine the new value to use in the resulting dict.
 *
 * ## Examples
 *
 * ```gleam
 * let a = from_list([#("a", 0), #("b", 1)])
 * let b = from_list([#("a", 2), #("c", 3)])
 * assert combine(a, b, fn(one, other) { one + other })
 *   == from_list([#("a", 2), #("b", 1), #("c", 3)])
 * ```
 */ parcelHelpers.export(exports, "combine", ()=>combine);
/**
 * Creates a new dict from a pair of given dicts by combining their entries.
 *
 * If there are entries with the same keys in both dicts the entry from the
 * second dict takes precedence.
 *
 * ## Examples
 *
 * ```gleam
 * let a = from_list([#("a", 0), #("b", 1)])
 * let b = from_list([#("b", 2), #("c", 3)])
 * assert merge(a, b) == from_list([#("a", 0), #("b", 2), #("c", 3)])
 * ```
 */ parcelHelpers.export(exports, "merge", ()=>merge);
parcelHelpers.export(exports, "group", ()=>group);
var _dictMjs = require("../dict.mjs");
var _gleamMjs = require("../gleam.mjs");
var _optionMjs = require("../gleam/option.mjs");
function is_empty(dict) {
    return (0, _dictMjs.size)(dict) === 0;
}
function from_list_loop(loop$transient, loop$list) {
    while(true){
        let transient = loop$transient;
        let list = loop$list;
        if (list instanceof (0, _gleamMjs.Empty)) return (0, _dictMjs.fromTransient)(transient);
        else {
            let rest = list.tail;
            let key = list.head[0];
            let value = list.head[1];
            loop$transient = (0, _dictMjs.destructiveTransientInsert)(key, value, transient);
            loop$list = rest;
        }
    }
}
function from_list(list) {
    return from_list_loop((0, _dictMjs.toTransient)((0, _dictMjs.make)()), list);
}
function do_take_loop(loop$dict, loop$desired_keys, loop$acc) {
    while(true){
        let dict = loop$dict;
        let desired_keys = loop$desired_keys;
        let acc = loop$acc;
        if (desired_keys instanceof (0, _gleamMjs.Empty)) return (0, _dictMjs.fromTransient)(acc);
        else {
            let key = desired_keys.head;
            let rest = desired_keys.tail;
            let $ = (0, _dictMjs.get)(dict, key);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let value = $[0];
                loop$dict = dict;
                loop$desired_keys = rest;
                loop$acc = (0, _dictMjs.destructiveTransientInsert)(key, value, acc);
            } else {
                loop$dict = dict;
                loop$desired_keys = rest;
                loop$acc = acc;
            }
        }
    }
}
function do_take(desired_keys, dict) {
    return do_take_loop(dict, desired_keys, (0, _dictMjs.toTransient)((0, _dictMjs.make)()));
}
function take(dict, desired_keys) {
    return do_take(desired_keys, dict);
}
function delete$(dict, key) {
    let _pipe = (0, _dictMjs.toTransient)(dict);
    let _pipe$1 = ((_capture)=>{
        return (0, _dictMjs.destructiveTransientDelete)(key, _capture);
    })(_pipe);
    return (0, _dictMjs.fromTransient)(_pipe$1);
}
function drop_loop(loop$transient, loop$disallowed_keys) {
    while(true){
        let transient = loop$transient;
        let disallowed_keys = loop$disallowed_keys;
        if (disallowed_keys instanceof (0, _gleamMjs.Empty)) return (0, _dictMjs.fromTransient)(transient);
        else {
            let key = disallowed_keys.head;
            let rest = disallowed_keys.tail;
            loop$transient = (0, _dictMjs.destructiveTransientDelete)(key, transient);
            loop$disallowed_keys = rest;
        }
    }
}
function do_drop(disallowed_keys, dict) {
    return drop_loop((0, _dictMjs.toTransient)(dict), disallowed_keys);
}
function drop(dict, disallowed_keys) {
    return do_drop(disallowed_keys, dict);
}
function upsert(dict, key, fun) {
    let $ = (0, _dictMjs.get)(dict, key);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let value = $[0];
        return (0, _dictMjs.insert)(dict, key, fun(new _optionMjs.Some(value)));
    } else return (0, _dictMjs.insert)(dict, key, fun(new _optionMjs.None()));
}
function to_list(dict) {
    return (0, _dictMjs.fold)(dict, (0, _gleamMjs.toList)([]), (acc, key, value)=>{
        return (0, _gleamMjs.prepend)([
            key,
            value
        ], acc);
    });
}
function keys(dict) {
    return (0, _dictMjs.fold)(dict, (0, _gleamMjs.toList)([]), (acc, key, _)=>{
        return (0, _gleamMjs.prepend)(key, acc);
    });
}
function values(dict) {
    return (0, _dictMjs.fold)(dict, (0, _gleamMjs.toList)([]), (acc, _, value)=>{
        return (0, _gleamMjs.prepend)(value, acc);
    });
}
function do_filter(f, dict) {
    let _pipe = (0, _dictMjs.toTransient)((0, _dictMjs.make)());
    let _pipe$1 = (0, _dictMjs.fold)(dict, _pipe, (transient, key, value)=>{
        let $ = f(key, value);
        if ($) return (0, _dictMjs.destructiveTransientInsert)(key, value, transient);
        else return transient;
    });
    return (0, _dictMjs.fromTransient)(_pipe$1);
}
function filter(dict, predicate) {
    return do_filter(predicate, dict);
}
function each(dict, fun) {
    return (0, _dictMjs.fold)(dict, undefined, (nil, k, v)=>{
        fun(k, v);
        return nil;
    });
}
function do_combine(combine, left, right) {
    let _block;
    let $1 = (0, _dictMjs.size)(left) >= (0, _dictMjs.size)(right);
    if ($1) _block = [
        left,
        right,
        combine
    ];
    else _block = [
        right,
        left,
        (k, l, r)=>{
            return combine(k, r, l);
        }
    ];
    let $ = _block;
    let big;
    let small;
    let combine$1;
    big = $[0];
    small = $[1];
    combine$1 = $[2];
    let _pipe = (0, _dictMjs.toTransient)(big);
    let _pipe$1 = (0, _dictMjs.fold)(small, _pipe, (transient, key, value)=>{
        let update = (existing)=>{
            return combine$1(key, existing, value);
        };
        return (0, _dictMjs.destructiveTransientUpdateWith)(key, update, value, transient);
    });
    return (0, _dictMjs.fromTransient)(_pipe$1);
}
function combine(dict, other, fun) {
    return do_combine((_, l, r)=>{
        return fun(l, r);
    }, dict, other);
}
function merge(dict, new_entries) {
    return combine(dict, new_entries, (_, new_entry)=>{
        return new_entry;
    });
}
function group_loop(loop$transient, loop$to_key, loop$list) {
    while(true){
        let transient = loop$transient;
        let to_key = loop$to_key;
        let list = loop$list;
        if (list instanceof (0, _gleamMjs.Empty)) return (0, _dictMjs.fromTransient)(transient);
        else {
            let value = list.head;
            let rest = list.tail;
            let key = to_key(value);
            let update = (existing)=>{
                return (0, _gleamMjs.prepend)(value, existing);
            };
            let _pipe = transient;
            let _pipe$1 = ((_capture)=>{
                return (0, _dictMjs.destructiveTransientUpdateWith)(key, update, (0, _gleamMjs.toList)([
                    value
                ]), _capture);
            })(_pipe);
            loop$transient = _pipe$1;
            loop$to_key = to_key;
            loop$list = rest;
        }
    }
}
function group(key, list) {
    return group_loop((0, _dictMjs.toTransient)((0, _dictMjs.make)()), key, list);
}

},{"../dict.mjs":"287yP","../gleam.mjs":"aiPrb","../gleam/option.mjs":"aWtoH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"287yP":[function(require,module,exports,__globalThis) {
/**
 * This file uses jsdoc to annotate types.
 * These types can be checked using the typescript compiler with "checkjs" option.
 */ var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * hash any js value
 * @param {any} u
 * @returns {number}
 */ parcelHelpers.export(exports, "getHash", ()=>getHash);
parcelHelpers.export(exports, "make", ()=>make);
parcelHelpers.export(exports, "from", ()=>from);
parcelHelpers.export(exports, "size", ()=>size);
parcelHelpers.export(exports, "get", ()=>get);
parcelHelpers.export(exports, "has", ()=>has);
/**
 * We use "transient" values to allow for safer internal mutations of the data
 * structure. This is an optimisation only. No mutable API is exposed to the user.
 *
 * Transients are to be treated as having a linear (single-use, think rust) type.
 * A transient value becomes invalid as soon as it's passed to one of the functions.
 *
 * Internally, we track a "generation" value on each node. If the generation
 * doesn't match the one for the current transient, we have to copy - the node
 * could still be referenced by another dict instance!
 * After that, no other references than the transient one exists, so it's safe
 * to mutate in place.
 */ parcelHelpers.export(exports, "toTransient", ()=>toTransient);
/**
 * Consume a transient, producing a normal Dict again.
 */ parcelHelpers.export(exports, "fromTransient", ()=>fromTransient);
parcelHelpers.export(exports, "insert", ()=>insert);
/**
 * Consume a transient, writing a new key/value pair into the dictionary it
 * represents. If the key already exists, it will be overwritten.
 *
 * Returns a new transient.
 */ parcelHelpers.export(exports, "destructiveTransientInsert", ()=>destructiveTransientInsert);
/**
 * Consume a transient, writing a new key/value pair if the key doesn't exist or updating
 * the existing value with a function if it does.
 *
 * Returns a new transient.
 */ parcelHelpers.export(exports, "destructiveTransientUpdateWith", ()=>destructiveTransientUpdateWith);
/**
 * Consume a transient, removing a key if it exists.
 * Returns a new transient.
 */ parcelHelpers.export(exports, "destructiveTransientDelete", ()=>destructiveTransientDelete);
parcelHelpers.export(exports, "map", ()=>map);
parcelHelpers.export(exports, "fold", ()=>fold);
var _gleamMjs = require("./gleam.mjs");
// -- HASH --------------------------------------------------------------------
const referenceMap = /* @__PURE__ */ new WeakMap();
const tempDataView = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
let referenceUID = 0;
/**
 * hash the object by reference using a weak map and incrementing uid
 * @param {any} o
 * @returns {number}
 */ function hashByReference(o) {
    const known = referenceMap.get(o);
    if (known !== undefined) return known;
    const hash = referenceUID++;
    if (referenceUID === 0x7fffffff) referenceUID = 0;
    referenceMap.set(o, hash);
    return hash;
}
/**
 * merge two hashes in an order sensitive way
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */ function hashMerge(a, b) {
    return a ^ b + 0x9e3779b9 + (a << 6) + (a >> 2) | 0;
}
/**
 * standard string hash popularised by Java
 * @param {string} s
 * @returns {number}
 */ function hashString(s) {
    let hash = 0;
    const len = s.length;
    for(let i = 0; i < len; i++)hash = Math.imul(31, hash) + s.charCodeAt(i) | 0;
    return hash;
}
/**
 * hash a number by converting to two integers and do some jumbling
 * @param {number} n
 * @returns {number}
 */ function hashNumber(n) {
    tempDataView.setFloat64(0, n);
    const i = tempDataView.getInt32(0);
    const j = tempDataView.getInt32(4);
    return Math.imul(0x45d9f3b, i >> 16 ^ i) ^ j;
}
/**
 * hash a BigInt by converting it to a string and hashing that
 * @param {BigInt} n
 * @returns {number}
 */ function hashBigInt(n) {
    return hashString(n.toString());
}
/**
 * hash any js object
 * @param {any} o
 * @returns {number}
 */ function hashObject(o) {
    const proto = Object.getPrototypeOf(o);
    if (proto !== null && typeof proto.hashCode === "function") try {
        const code = o.hashCode(o);
        if (typeof code === "number") return code;
    } catch  {}
    if (o instanceof Promise || o instanceof WeakSet || o instanceof WeakMap) return hashByReference(o);
    if (o instanceof Date) return hashNumber(o.getTime());
    let h = 0;
    if (o instanceof ArrayBuffer) o = new Uint8Array(o);
    if (Array.isArray(o) || o instanceof Uint8Array) for(let i = 0; i < o.length; i++)h = Math.imul(31, h) + getHash(o[i]) | 0;
    else if (o instanceof Set) o.forEach((v)=>{
        h = h + getHash(v) | 0;
    });
    else if (o instanceof Map) o.forEach((v, k)=>{
        h = h + hashMerge(getHash(v), getHash(k)) | 0;
    });
    else {
        const keys = Object.keys(o);
        for(let i = 0; i < keys.length; i++){
            const k = keys[i];
            const v = o[k];
            h = h + hashMerge(getHash(v), hashString(k)) | 0;
        }
    }
    return h;
}
function getHash(u) {
    if (u === null) return 0x42108422;
    if (u === undefined) return 0x42108423;
    if (u === true) return 0x42108421;
    if (u === false) return 0x42108420;
    switch(typeof u){
        case "number":
            return hashNumber(u);
        case "string":
            return hashString(u);
        case "bigint":
            return hashBigInt(u);
        case "object":
            return hashObject(u);
        case "symbol":
            return hashByReference(u);
        case "function":
            return hashByReference(u);
        default:
            return 0; // should be unreachable
    }
}
class Dict {
    constructor(size, root){
        this.size = size;
        this.root = root;
    }
}
exports.default = Dict;
/// The power-of-2 branching factor for the dict. For example, a value of `5` indicates a 32-ary tree.
const bits = 5;
const mask = (1 << bits) - 1;
/// This symbol is used internally to avoid constructing results.
const noElementMarker = Symbol();
/// This symbol is used to store the "generation" on a node.
/// Using a symbol makes the property not enumerable, which means the generation
/// will be ignored during equality checks.
const generationKey = Symbol();
// Some commonly used constants throughout the code.
const emptyNode = /* @__PURE__ */ newNode(0);
const emptyDict = /* @__PURE__ */ new Dict(0, emptyNode);
const errorNil = /* @__PURE__ */ (0, _gleamMjs.Result$Error)(undefined);
function makeNode(generation, datamap, nodemap, data1) {
    // The order of fields is important, as they define the order `isEqual` will
    // compare our fields. Putting the bitmaps first means that equality can
    // early-out if the bitmaps are not equal.
    return {
        // A node is a high-arity (32 in practice) hybrid tree node.
        // Hybrid means that it stores data directly as well as pointers to child nodes.
        //
        // Each node contains 2 bitmaps:
        // - The datamap has a bit set if that slot in the node contains direct data
        // - The nodemap has a bit set if that slot in the node contains another node.
        //
        // Both are exclusive to on another, so datamap & nodemap == 0.
        //
        // Every key/hash value directly correlates to a specific bit by using a trie
        // suffix (least significant bits first) encoding.
        // For example, if the last 5 bits of the hash are 1101, the bit to check for
        // that value is the 13th bit.
        datamap,
        nodemap,
        // The slots itself are stored in a single contiguous array that contains
        // both direct k/v-pairs and child nodes.
        //
        // The direct children come first, followed by the child nodes in _reverse order_:
        //
        //              7654321
        //     datamap: 1000100
        //     nodemap:   10011
        //     data: [key3, value3, key7, value7, child5, child2, child1]
        //            ------------------------->  <---------------------
        //                     datamap                    nodemap
        //
        // Every `1` bit in the datamap corresponds to a pair of [key, value] entries,
        // and every `1` bit in the nodemap corresponds to a child node entry.
        //
        // Children are stored in reverse order to avoid having to store or calculate an
        // "offset" value to skip over the direct children.
        data: data1,
        // The generation is used to track which nodes need to be copied during transient updates.
        // Using a symbol here makes `isEqual` ignore this field.
        [generationKey]: generation
    };
}
function newNode(generation) {
    return makeNode(generation, 0, 0, []);
}
/**
 * Copies a node and its data array if it's from another generation, making it safe
 * to mutate the node.
 */ function copyNode(node, generation) {
    if (node[generationKey] === generation) return node;
    const newData = node.data.slice(0);
    return makeNode(generation, node.datamap, node.nodemap, newData);
}
/**
 * Copies a node if needed and sets a new value.
 */ function copyAndSet(node, generation, idx, val) {
    if (node.data[idx] === val) return node;
    // Using copyNode is faster than a specialised implementation.
    node = copyNode(node, generation);
    node.data[idx] = val;
    return node;
}
/**
 * Copies a node if needed, and then inserts a new key-value pair.
 */ function copyAndInsertPair(node, generation, bit, idx, key, val) {
    const data1 = node.data;
    const length = data1.length;
    // the fastest way to insert a pair is to always copy.
    const newData = new Array(length + 2);
    let readIndex = 0;
    let writeIndex = 0;
    while(readIndex < idx)newData[writeIndex++] = data1[readIndex++];
    newData[writeIndex++] = key;
    newData[writeIndex++] = val;
    while(readIndex < length)newData[writeIndex++] = data1[readIndex++];
    return makeNode(generation, node.datamap | bit, node.nodemap, newData);
}
function copyAndRemovePair(node, generation, bit, idx) {
    node = copyNode(node, generation);
    const data1 = node.data;
    const length = data1.length;
    for(let w = idx, r = idx + 2; r < length; ++r, ++w)data1[w] = data1[r];
    data1.pop();
    data1.pop();
    node.datamap ^= bit;
    return node;
}
function make() {
    return emptyDict;
}
function from(iterable) {
    let transient = toTransient(emptyDict);
    for (const [key, value] of iterable)transient = destructiveTransientInsert(key, value, transient);
    return fromTransient(transient);
}
function size(dict) {
    return dict.size;
}
function get(dict, key) {
    const result = lookup(dict.root, key, getHash(key));
    return result !== noElementMarker ? (0, _gleamMjs.Result$Ok)(result) : errorNil;
}
function has(dict, key) {
    return lookup(dict.root, key, getHash(key)) !== noElementMarker;
}
function lookup(node, key, hash) {
    for(let shift = 0; shift < 32; shift += bits){
        const data1 = node.data;
        const bit = hashbit(hash, shift);
        if (node.nodemap & bit) // we found our hash inside the nodemap, so we can continue our search there.
        node = data1[data1.length - 1 - index(node.nodemap, bit)];
        else if (node.datamap & bit) {
            // we store this hash directly!
            //
            // this also means that there are no other values with the same
            // hash prefix in this dict.
            //
            // We still need to check if the key matches, but if it does we know for
            // sure that this is the correct value, and if it doesn't that we don't
            // contain the value in question.
            const dataidx = Math.imul(index(node.datamap, bit), 2);
            return (0, _gleamMjs.isEqual)(key, data1[dataidx]) ? data1[dataidx + 1] : noElementMarker;
        } else // if the hash bit is not set in neither bitmaps, we immediately know that
        // this key cannot be inside this dict.
        return noElementMarker;
    }
    // our shift has exceeded 32 bits. Everything that follows is
    // implicitely an overflow node and only contains direct children.
    const overflow = node.data;
    for(let i = 0; i < overflow.length; i += 2){
        if ((0, _gleamMjs.isEqual)(key, overflow[i])) return overflow[i + 1];
    }
    return noElementMarker;
}
function toTransient(dict) {
    return {
        generation: nextGeneration(dict),
        root: dict.root,
        size: dict.size,
        dict: dict
    };
}
function fromTransient(transient) {
    if (transient.root === transient.dict.root) return transient.dict;
    return new Dict(transient.size, transient.root);
}
/**
 * Find and allocate the next generation id.
 *
 * @template K,V
 * @param {Dict<K,V>} dict
 * @returns {number}
 */ function nextGeneration(dict) {
    const root = dict.root;
    if (root[generationKey] < Number.MAX_SAFE_INTEGER) return root[generationKey] + 1;
    // we have reached MAX_SAFE_INTEGER generations -
    // at this point, we have to walk the dictionary once to reset the counter
    // on every node. This is safe since it's part of the contract for transient
    // that only one of them exists at any given time.
    //
    const queue = [
        root
    ];
    while(queue.length){
        // order doesn't matter, so we can use push/pop for faster array usage.
        const node = queue.pop();
        // reset the generation to 0
        node[generationKey] = 0;
        // queue all other referenced nodes
        // We need to query the length from the nodemap, as we don't know if this
        //  is an overflow node or not! if it is, it will never have datamap set!
        const nodeStart = data.length - popcount(node.nodemap);
        for(let i = nodeStart; i < node.data.length; ++i)queue.push(node.data[i]);
    }
    return 1;
}
/// Insert is the second-most performance-sensitive operation.
/// We use a global "transient" value here to avoid doing a memory allocation.
const globalTransient = /* @__PURE__ */ toTransient(emptyDict);
function insert(dict, key, value) {
    globalTransient.generation = nextGeneration(dict);
    globalTransient.size = dict.size;
    const hash = getHash(key);
    const root = insertIntoNode(globalTransient, dict.root, key, value, hash, 0);
    if (root === dict.root) return dict;
    return new Dict(globalTransient.size, root);
}
function destructiveTransientInsert(key, value, transient) {
    const hash = getHash(key);
    transient.root = insertIntoNode(transient, transient.root, key, value, hash, 0);
    return transient;
}
function destructiveTransientUpdateWith(key, fun, value, transient) {
    const hash = getHash(key);
    const existing = lookup(transient.root, key, hash);
    if (existing !== noElementMarker) value = fun(existing);
    transient.root = insertIntoNode(transient, transient.root, key, value, hash, 0);
    return transient;
}
function insertIntoNode(transient, node, key, value, hash, shift) {
    const data1 = node.data;
    const generation = transient.generation;
    // 1. Overflow Node
    // overflow nodes only contain key/value-pairs. we walk the data linearly trying to find a match.
    if (shift > 32) {
        for(let i = 0; i < data1.length; i += 2){
            if ((0, _gleamMjs.isEqual)(key, data1[i])) return copyAndSet(node, generation, i + 1, value);
        }
        transient.size += 1;
        return copyAndInsertPair(node, generation, 0, data1.length, key, value);
    }
    const bit = hashbit(hash, shift);
    // 2. Child Node
    // We have to check first if there is already a child node we have to traverse to.
    if (node.nodemap & bit) {
        const nodeidx = data1.length - 1 - index(node.nodemap, bit);
        let child = data1[nodeidx];
        child = insertIntoNode(transient, child, key, value, hash, shift + bits);
        return copyAndSet(node, generation, nodeidx, child);
    }
    // 3. New Data Node
    // No child node and no data node exists yet, so we can potentially just insert a new value.
    const dataidx = Math.imul(index(node.datamap, bit), 2);
    if ((node.datamap & bit) === 0) {
        transient.size += 1;
        return copyAndInsertPair(node, generation, bit, dataidx, key, value);
    }
    // 4. Existing Data Node
    // We have a match that we can update, or remove.
    if ((0, _gleamMjs.isEqual)(key, data1[dataidx])) return copyAndSet(node, generation, dataidx + 1, value);
    // 5. Collision
    // There is no child node, but a data node with the same hash, but with a different key.
    // To resolve this, we push both nodes down one level.
    const childShift = shift + bits;
    let child = emptyNode;
    child = insertIntoNode(transient, child, key, value, hash, childShift);
    const key2 = data1[dataidx];
    const value2 = data1[dataidx + 1];
    const hash2 = getHash(key2);
    child = insertIntoNode(transient, child, key2, value2, hash2, childShift);
    // we inserted 2 elements, but implicitely deleted the one we pushed down from the datamap.
    transient.size -= 1;
    // remove the old data pair, and insert the new child node.
    const length = data1.length;
    const nodeidx = length - 1 - index(node.nodemap, bit);
    // writing these loops in javascript instead of a combination of splices
    // turns out to be faster. Copying always turned out to be faster.
    const newData = new Array(length - 1);
    let readIndex = 0;
    let writeIndex = 0;
    // [0..dataidx, skip 2 elements, ..nodeidx, newChild, ..rest]
    while(readIndex < dataidx)newData[writeIndex++] = data1[readIndex++];
    readIndex += 2;
    while(readIndex <= nodeidx)newData[writeIndex++] = data1[readIndex++];
    newData[writeIndex++] = child;
    while(readIndex < length)newData[writeIndex++] = data1[readIndex++];
    return makeNode(generation, node.datamap ^ bit, node.nodemap | bit, newData);
}
function destructiveTransientDelete(key, transient) {
    const hash = getHash(key);
    transient.root = deleteFromNode(transient, transient.root, key, hash, 0);
    return transient;
}
function deleteFromNode(transient, node, key, hash, shift) {
    const data1 = node.data;
    const generation = transient.generation;
    // 1. Overflow Node
    // overflow nodes only contain key/value-pairs. we walk the data linearly trying to find a match.
    if (shift > 32) {
        for(let i = 0; i < data1.length; i += 2)if ((0, _gleamMjs.isEqual)(key, data1[i])) {
            transient.size -= 1;
            return copyAndRemovePair(node, generation, 0, i);
        }
        return node;
    }
    const bit = hashbit(hash, shift);
    const dataidx = Math.imul(index(node.datamap, bit), 2);
    // 2. Child Node
    // We have to check first if there is already a child node we have to traverse to.
    if ((node.nodemap & bit) !== 0) {
        const nodeidx = data1.length - 1 - index(node.nodemap, bit);
        let child = data1[nodeidx];
        child = deleteFromNode(transient, child, key, hash, shift + bits);
        // the node did change, so let's copy to incorporate that change.
        if (child.nodemap !== 0 || child.data.length > 2) return copyAndSet(node, generation, nodeidx, child);
        // this node only has a single data (k/v-pair) child.
        // to restore the CHAMP invariant, we "pull" that pair up into ourselves.
        // this ensures that every tree stays in its single optimal representation,
        // and allows dicts to be structurally compared.
        const length = data1.length;
        const newData = new Array(length + 1);
        let readIndex = 0;
        let writeIndex = 0;
        while(readIndex < dataidx)newData[writeIndex++] = data1[readIndex++];
        newData[writeIndex++] = child.data[0];
        newData[writeIndex++] = child.data[1];
        while(readIndex < nodeidx)newData[writeIndex++] = data1[readIndex++];
        readIndex++;
        while(readIndex < length)newData[writeIndex++] = data1[readIndex++];
        return makeNode(generation, node.datamap | bit, node.nodemap ^ bit, newData);
    }
    // 3. Data Node
    // There is no data entry here, or it is a prefix for a different key
    if ((node.datamap & bit) === 0 || !(0, _gleamMjs.isEqual)(key, data1[dataidx])) return node;
    // we found a data entry that we can delete.
    transient.size -= 1;
    return copyAndRemovePair(node, generation, bit, dataidx);
}
function map(dict, fun) {
    // map can never modify the structure, so we can walk the dictionary directly,
    // but still move to a new generation to make sure we get a new copy of every node.
    const generation = nextGeneration(dict);
    const root = copyNode(dict.root, generation);
    const queue = [
        root
    ];
    while(queue.length){
        // order doesn't matter, so we can use push/pop for faster array usage.
        const node = queue.pop();
        const data1 = node.data;
        // every node contains popcount(datamap) direct entries
        // We need to query the length from the nodemap, as we don't know if this
        //  is an overflow node or not! if it is, it will never have datamap set!
        const edgesStart = data1.length - popcount(node.nodemap);
        for(let i = 0; i < edgesStart; i += 2)// we copied the node while queueing it, so direct mutation here is safe.
        data1[i + 1] = fun(data1[i], data1[i + 1]);
        // the remaining entries are other nodes we can queue
        for(let i = edgesStart; i < data1.length; ++i){
            // copy the node first to make it safe to mutate
            data1[i] = copyNode(data1[i], generation);
            queue.push(data1[i]);
        }
    }
    return new Dict(dict.size, root);
}
function fold(dict, state, fun) {
    const queue = [
        dict.root
    ];
    while(queue.length){
        // order doesn't matter, so we can use push/pop for faster array usage.
        const node = queue.pop();
        const data1 = node.data;
        // every node contains popcount(datamap) direct entries
        // We need to query the length from the nodemap, as we don't know if this
        //  is an overflow node or not! if it is, it will never have datamap set!
        const edgesStart = data1.length - popcount(node.nodemap);
        for(let i = 0; i < edgesStart; i += 2)state = fun(state, data1[i], data1[i + 1]);
        // the remaining entries are child nodes we can queue.
        for(let i = edgesStart; i < data1.length; ++i)queue.push(data1[i]);
    }
    return state;
}
/**
 * How many `1` bits are set in a 32-bit integer.
 */ function popcount(n) {
    n -= n >>> 1 & 0x55555555;
    n = (n & 0x33333333) + (n >>> 2 & 0x33333333);
    return Math.imul(n + (n >>> 4) & 0x0f0f0f0f, 0x01010101) >>> 24;
}
/**
 * Given a population bitmap and a bit selected from that map, returns
 * how many less significant 1 bits there are.
 *
 * For example, index(10101, 100) returns 1, since there is a single less
 * significant `1` bit. This translates to the 0-based "index" of that bit.
 */ function index(bitmap, bit) {
    return popcount(bitmap & bit - 1);
}
/**
 * Extracts a single slice of the hash, and returns a bitmask for the resulting value.
 * For example, if the slice returns 5, this function returns 10000 = 1 << 5.
 */ function hashbit(hash, shift) {
    return 1 << (hash >>> shift & mask);
}

},{"./gleam.mjs":"aiPrb","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"aWtoH":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Some", ()=>Some);
parcelHelpers.export(exports, "Option$Some", ()=>Option$Some);
parcelHelpers.export(exports, "Option$isSome", ()=>Option$isSome);
parcelHelpers.export(exports, "Option$Some$0", ()=>Option$Some$0);
parcelHelpers.export(exports, "None", ()=>None);
parcelHelpers.export(exports, "Option$None", ()=>Option$None);
parcelHelpers.export(exports, "Option$isNone", ()=>Option$isNone);
/**
 * Combines a list of `Option`s into a single `Option`.
 * If all elements in the list are `Some` then returns a `Some` holding the list of values.
 * If any element is `None` then returns `None`.
 *
 * ## Examples
 *
 * ```gleam
 * assert all([Some(1), Some(2)]) == Some([1, 2])
 * ```
 *
 * ```gleam
 * assert all([Some(1), None]) == None
 * ```
 */ parcelHelpers.export(exports, "all", ()=>all);
/**
 * Checks whether the `Option` is a `Some` value.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_some(Some(1))
 * ```
 *
 * ```gleam
 * assert !is_some(None)
 * ```
 */ parcelHelpers.export(exports, "is_some", ()=>is_some);
/**
 * Checks whether the `Option` is a `None` value.
 *
 * ## Examples
 *
 * ```gleam
 * assert !is_none(Some(1))
 * ```
 *
 * ```gleam
 * assert is_none(None)
 * ```
 */ parcelHelpers.export(exports, "is_none", ()=>is_none);
/**
 * Converts an `Option` type to a `Result` type.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_result(Some(1), "some_error") == Ok(1)
 * ```
 *
 * ```gleam
 * assert to_result(None, "some_error") == Error("some_error")
 * ```
 */ parcelHelpers.export(exports, "to_result", ()=>to_result);
/**
 * Converts a `Result` type to an `Option` type.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_result(Ok(1)) == Some(1)
 * ```
 *
 * ```gleam
 * assert from_result(Error("some_error")) == None
 * ```
 */ parcelHelpers.export(exports, "from_result", ()=>from_result);
/**
 * Extracts the value from an `Option`, returning a default value if there is none.
 *
 * ## Examples
 *
 * ```gleam
 * assert unwrap(Some(1), 0) == 1
 * ```
 *
 * ```gleam
 * assert unwrap(None, 0) == 0
 * ```
 */ parcelHelpers.export(exports, "unwrap", ()=>unwrap);
/**
 * Extracts the value from an `Option`, evaluating the default function if the option is `None`.
 *
 * ## Examples
 *
 * ```gleam
 * assert lazy_unwrap(Some(1), fn() { 0 }) == 1
 * ```
 *
 * ```gleam
 * assert lazy_unwrap(None, fn() { 0 }) == 0
 * ```
 */ parcelHelpers.export(exports, "lazy_unwrap", ()=>lazy_unwrap);
/**
 * Updates a value held within the `Some` of an `Option` by calling a given function
 * on it.
 *
 * If the `Option` is a `None` rather than `Some`, the function is not called and the
 * `Option` stays the same.
 *
 * ## Examples
 *
 * ```gleam
 * assert map(over: Some(1), with: fn(x) { x + 1 }) == Some(2)
 * ```
 *
 * ```gleam
 * assert map(over: None, with: fn(x) { x + 1 }) == None
 * ```
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Merges a nested `Option` into a single layer.
 *
 * ## Examples
 *
 * ```gleam
 * assert flatten(Some(Some(1))) == Some(1)
 * ```
 *
 * ```gleam
 * assert flatten(Some(None)) == None
 * ```
 *
 * ```gleam
 * assert flatten(None) == None
 * ```
 */ parcelHelpers.export(exports, "flatten", ()=>flatten);
/**
 * Updates a value held within the `Some` of an `Option` by calling a given function
 * on it, where the given function also returns an `Option`. The two options are
 * then merged together into one `Option`.
 *
 * If the `Option` is a `None` rather than `Some` the function is not called and the
 * option stays the same.
 *
 * This function is the equivalent of calling `map` followed by `flatten`, and
 * it is useful for chaining together multiple functions that return `Option`.
 *
 * ## Examples
 *
 * ```gleam
 * assert then(Some(1), fn(x) { Some(x + 1) }) == Some(2)
 * ```
 *
 * ```gleam
 * assert then(Some(1), fn(x) { Some(#("a", x)) }) == Some(#("a", 1))
 * ```
 *
 * ```gleam
 * assert then(Some(1), fn(_) { None }) == None
 * ```
 *
 * ```gleam
 * assert then(None, fn(x) { Some(x + 1) }) == None
 * ```
 */ parcelHelpers.export(exports, "then$", ()=>then$);
/**
 * Returns the first value if it is `Some`, otherwise returns the second value.
 *
 * ## Examples
 *
 * ```gleam
 * assert or(Some(1), Some(2)) == Some(1)
 * ```
 *
 * ```gleam
 * assert or(Some(1), None) == Some(1)
 * ```
 *
 * ```gleam
 * assert or(None, Some(2)) == Some(2)
 * ```
 *
 * ```gleam
 * assert or(None, None) == None
 * ```
 */ parcelHelpers.export(exports, "or", ()=>or);
/**
 * Returns the first value if it is `Some`, otherwise evaluates the given function for a fallback value.
 *
 * ## Examples
 *
 * ```gleam
 * assert lazy_or(Some(1), fn() { Some(2) }) == Some(1)
 * ```
 *
 * ```gleam
 * assert lazy_or(Some(1), fn() { None }) == Some(1)
 * ```
 *
 * ```gleam
 * assert lazy_or(None, fn() { Some(2) }) == Some(2)
 * ```
 *
 * ```gleam
 * assert lazy_or(None, fn() { None }) == None
 * ```
 */ parcelHelpers.export(exports, "lazy_or", ()=>lazy_or);
/**
 * Given a list of `Option`s,
 * returns only the values inside `Some`.
 *
 * ## Examples
 *
 * ```gleam
 * assert values([Some(1), None, Some(3)]) == [1, 3]
 * ```
 */ parcelHelpers.export(exports, "values", ()=>values);
var _gleamMjs = require("../gleam.mjs");
class Some extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const Option$Some = ($0)=>new Some($0);
const Option$isSome = (value)=>value instanceof Some;
const Option$Some$0 = (value)=>value[0];
class None extends (0, _gleamMjs.CustomType) {
}
const Option$None = ()=>new None();
const Option$isNone = (value)=>value instanceof None;
function reverse_and_prepend(loop$prefix, loop$suffix) {
    while(true){
        let prefix = loop$prefix;
        let suffix = loop$suffix;
        if (prefix instanceof (0, _gleamMjs.Empty)) return suffix;
        else {
            let first = prefix.head;
            let rest = prefix.tail;
            loop$prefix = rest;
            loop$suffix = (0, _gleamMjs.prepend)(first, suffix);
        }
    }
}
function reverse(list) {
    return reverse_and_prepend(list, (0, _gleamMjs.toList)([]));
}
function all_loop(loop$list, loop$acc) {
    while(true){
        let list = loop$list;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return new Some(reverse(acc));
        else {
            let $ = list.head;
            if ($ instanceof Some) {
                let rest = list.tail;
                let first = $[0];
                loop$list = rest;
                loop$acc = (0, _gleamMjs.prepend)(first, acc);
            } else return new None();
        }
    }
}
function all(list) {
    return all_loop(list, (0, _gleamMjs.toList)([]));
}
function is_some(option) {
    return !(option instanceof None);
}
function is_none(option) {
    return option instanceof None;
}
function to_result(option, e) {
    if (option instanceof Some) {
        let a = option[0];
        return new (0, _gleamMjs.Ok)(a);
    } else return new (0, _gleamMjs.Error)(e);
}
function from_result(result) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let a = result[0];
        return new Some(a);
    } else return new None();
}
function unwrap(option, default$) {
    if (option instanceof Some) {
        let x = option[0];
        return x;
    } else return default$;
}
function lazy_unwrap(option, default$) {
    if (option instanceof Some) {
        let x = option[0];
        return x;
    } else return default$();
}
function map(option, fun) {
    if (option instanceof Some) {
        let x = option[0];
        return new Some(fun(x));
    } else return option;
}
function flatten(option) {
    if (option instanceof Some) {
        let x = option[0];
        return x;
    } else return option;
}
function then$(option, fun) {
    if (option instanceof Some) {
        let x = option[0];
        return fun(x);
    } else return option;
}
function or(first, second) {
    if (first instanceof Some) return first;
    else return second;
}
function lazy_or(first, second) {
    if (first instanceof Some) return first;
    else return second();
}
function values_loop(loop$list, loop$acc) {
    while(true){
        let list = loop$list;
        let acc = loop$acc;
        if (list instanceof (0, _gleamMjs.Empty)) return reverse(acc);
        else {
            let $ = list.head;
            if ($ instanceof Some) {
                let rest = list.tail;
                let first = $[0];
                loop$list = rest;
                loop$acc = (0, _gleamMjs.prepend)(first, acc);
            } else {
                let rest = list.tail;
                loop$list = rest;
                loop$acc = acc;
            }
        }
    }
}
function values(options) {
    return values_loop(options, (0, _gleamMjs.toList)([]));
}

},{"../gleam.mjs":"aiPrb","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"9bPI9":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "ceiling", ()=>(0, _gleamStdlibMjs.ceiling));
parcelHelpers.export(exports, "exponential", ()=>(0, _gleamStdlibMjs.exp));
parcelHelpers.export(exports, "floor", ()=>(0, _gleamStdlibMjs.floor));
parcelHelpers.export(exports, "parse", ()=>(0, _gleamStdlibMjs.parse_float));
parcelHelpers.export(exports, "random", ()=>(0, _gleamStdlibMjs.random_uniform));
parcelHelpers.export(exports, "to_string", ()=>(0, _gleamStdlibMjs.float_to_string));
parcelHelpers.export(exports, "truncate", ()=>(0, _gleamStdlibMjs.truncate));
/**
 * Compares two `Float`s, returning an `Order`:
 * `Lt` for lower than, `Eq` for equals, or `Gt` for greater than.
 *
 * ## Examples
 *
 * ```gleam
 * assert compare(2.0, 2.3) == Lt
 * ```
 *
 * To handle
 * [Floating Point Imprecision](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems)
 * you may use [`loosely_compare`](#loosely_compare) instead.
 */ parcelHelpers.export(exports, "compare", ()=>compare);
/**
 * Compares two `Float`s, returning the smaller of the two.
 *
 * ## Examples
 *
 * ```gleam
 * assert min(2.0, 2.3) == 2.0
 * ```
 */ parcelHelpers.export(exports, "min", ()=>min);
/**
 * Compares two `Float`s, returning the larger of the two.
 *
 * ## Examples
 *
 * ```gleam
 * assert max(2.0, 2.3) == 2.3
 * ```
 */ parcelHelpers.export(exports, "max", ()=>max);
/**
 * Restricts a float between two bounds.
 *
 * Note: If the `min` argument is larger than the `max` argument then they
 * will be swapped, so the minimum bound is always lower than the maximum
 * bound.
 *
 *
 * ## Examples
 *
 * ```gleam
 * assert clamp(1.2, min: 1.4, max: 1.6) == 1.4
 * ```
 *
 * ```gleam
 * assert clamp(1.2, min: 1.4, max: 0.6) == 1.2
 * ```
 */ parcelHelpers.export(exports, "clamp", ()=>clamp);
/**
 * Returns the absolute value of the input as a `Float`.
 *
 * ## Examples
 *
 * ```gleam
 * assert absolute_value(-12.5) == 12.5
 * ```
 *
 * ```gleam
 * assert absolute_value(10.2) == 10.2
 * ```
 */ parcelHelpers.export(exports, "absolute_value", ()=>absolute_value);
/**
 * Compares two `Float`s within a tolerance, returning an `Order`:
 * `Lt` for lower than, `Eq` for equals, or `Gt` for greater than.
 *
 * This function allows Float comparison while handling
 * [Floating Point Imprecision](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems).
 *
 * Notice: For `Float`s the tolerance won't be exact:
 * `5.3 - 5.0` is not exactly `0.3`.
 *
 * ## Examples
 *
 * ```gleam
 * assert loosely_compare(5.0, with: 5.3, tolerating: 0.5) == Eq
 * ```
 *
 * If you want to check only for equality you may use
 * [`loosely_equals`](#loosely_equals) instead.
 */ parcelHelpers.export(exports, "loosely_compare", ()=>loosely_compare);
/**
 * Checks for equality of two `Float`s within a tolerance,
 * returning a `Bool`.
 *
 * This function allows Float comparison while handling
 * [Floating Point Imprecision](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems).
 *
 * Notice: For `Float`s the tolerance won't be exact:
 * `5.3 - 5.0` is not exactly `0.3`.
 *
 * ## Examples
 *
 * ```gleam
 * assert loosely_equals(5.0, with: 5.3, tolerating: 0.5)
 * ```
 *
 * ```gleam
 * assert !loosely_equals(5.0, with: 5.1, tolerating: 0.1)
 * ```
 */ parcelHelpers.export(exports, "loosely_equals", ()=>loosely_equals);
/**
 * Returns the result of the base being raised to the power of the
 * exponent, as a `Float`.
 *
 * ## Examples
 *
 * ```gleam
 * assert power(2.0, -1.0) == Ok(0.5)
 * ```
 *
 * ```gleam
 * assert power(2.0, 2.0) == Ok(4.0)
 * ```
 *
 * ```gleam
 * assert power(8.0, 1.5) == Ok(22.627416997969522)
 * ```
 *
 * ```gleam
 * assert 4.0 |> power(of: 2.0) == Ok(16.0)
 * ```
 *
 * ```gleam
 * assert power(-1.0, 0.5) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "power", ()=>power);
/**
 * Returns the square root of the input as a `Float`.
 *
 * ## Examples
 *
 * ```gleam
 * assert square_root(4.0) == Ok(2.0)
 * ```
 *
 * ```gleam
 * assert square_root(-16.0) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "square_root", ()=>square_root);
/**
 * Returns the negative of the value provided.
 *
 * ## Examples
 *
 * ```gleam
 * assert negate(1.0) == -1.0
 * ```
 */ parcelHelpers.export(exports, "negate", ()=>negate);
/**
 * Rounds the value to the nearest whole number as an `Int`.
 *
 * ## Examples
 *
 * ```gleam
 * assert round(2.3) == 2
 * ```
 *
 * ```gleam
 * assert round(2.5) == 3
 * ```
 */ parcelHelpers.export(exports, "round", ()=>round);
/**
 * Converts the value to a given precision as a `Float`.
 * The precision is the number of allowed decimal places.
 * Negative precisions are allowed and force rounding
 * to the nearest tenth, hundredth, thousandth etc.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_precision(2.43434348473, 2) == 2.43
 * ```
 *
 * ```gleam
 * assert to_precision(547890.453444, -3) == 548000.0
 * ```
 */ parcelHelpers.export(exports, "to_precision", ()=>to_precision);
/**
 * Sums a list of `Float`s.
 *
 * ## Example
 *
 * ```gleam
 * assert sum([1.0, 2.2, 3.3]) == 6.5
 * ```
 */ parcelHelpers.export(exports, "sum", ()=>sum);
/**
 * Multiplies a list of `Float`s and returns the product.
 *
 * ## Example
 *
 * ```gleam
 * assert product([2.5, 3.2, 4.2]) == 33.6
 * ```
 */ parcelHelpers.export(exports, "product", ()=>product);
/**
 * Computes the modulo of a float division of inputs as a `Result`.
 *
 * Returns division of the inputs as a `Result`: If the given divisor equals
 * `0`, this function returns an `Error`.
 *
 * The computed value will always have the same sign as the `divisor`.
 *
 * ## Examples
 *
 * ```gleam
 * assert modulo(13.3, by: 3.3) == Ok(0.1)
 * ```
 *
 * ```gleam
 * assert modulo(-13.3, by: 3.3) == Ok(3.2)
 * ```
 *
 * ```gleam
 * assert modulo(13.3, by: -3.3) == Ok(-3.2)
 * ```
 *
 * ```gleam
 * assert modulo(-13.3, by: -3.3) == Ok(-0.1)
 * ```
 */ parcelHelpers.export(exports, "modulo", ()=>modulo);
/**
 * Returns division of the inputs as a `Result`.
 *
 * ## Examples
 *
 * ```gleam
 * assert divide(0.0, 1.0) == Ok(0.0)
 * ```
 *
 * ```gleam
 * assert divide(1.0, 0.0) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "divide", ()=>divide);
/**
 * Adds two floats together.
 *
 * It's the function equivalent of the `+.` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert add(1.0, 2.0) == 3.0
 * ```
 *
 * ```gleam
 * import gleam/list
 *
 * assert list.fold([1.0, 2.0, 3.0], 0.0, add) == 6.0
 * ```
 *
 * ```gleam
 * assert 3.0 |> add(2.0) == 5.0
 * ```
 */ parcelHelpers.export(exports, "add", ()=>add);
/**
 * Multiplies two floats together.
 *
 * It's the function equivalent of the `*.` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert multiply(2.0, 4.0) == 8.0
 * ```
 *
 * ```gleam
 * import gleam/list
 *
 * assert list.fold([2.0, 3.0, 4.0], 1.0, multiply) == 24.0
 * ```
 *
 * ```gleam
 * assert 3.0 |> multiply(2.0) == 6.0
 * ```
 */ parcelHelpers.export(exports, "multiply", ()=>multiply);
/**
 * Subtracts one float from another.
 *
 * It's the function equivalent of the `-.` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert subtract(3.0, 1.0) == 2.0
 * ```
 *
 * ```gleam
 * import gleam/list
 *
 * assert list.fold([1.0, 2.0, 3.0], 10.0, subtract) == 4.0
 * ```
 *
 * ```gleam
 * assert 3.0 |> subtract(_, 2.0) == 1.0
 * ```
 *
 * ```gleam
 * assert 3.0 |> subtract(2.0, _) == -1.0
 * ```
 */ parcelHelpers.export(exports, "subtract", ()=>subtract);
/**
 * Returns the natural logarithm (base e) of the given `Float` as a `Result`. If the
 * input is less than or equal to 0, returns `Error(Nil)`.
 *
 * ## Examples
 *
 * ```gleam
 * assert logarithm(1.0) == Ok(0.0)
 * ```
 *
 * ```gleam
 * assert logarithm(2.718281828459045) == Ok(1.0)
 * ```
 *
 * ```gleam
 * assert logarithm(0.0) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert logarithm(-1.0) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "logarithm", ()=>logarithm);
var _gleamMjs = require("../gleam.mjs");
var _orderMjs = require("../gleam/order.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
function compare(a, b) {
    let $ = a === b;
    if ($) return new _orderMjs.Eq();
    else {
        let $1 = a < b;
        if ($1) return new _orderMjs.Lt();
        else return new _orderMjs.Gt();
    }
}
function min(a, b) {
    let $ = a < b;
    if ($) return a;
    else return b;
}
function max(a, b) {
    let $ = a > b;
    if ($) return a;
    else return b;
}
function clamp(x, min_bound, max_bound) {
    let $ = min_bound >= max_bound;
    if ($) {
        let _pipe = x;
        let _pipe$1 = min(_pipe, min_bound);
        return max(_pipe$1, max_bound);
    } else {
        let _pipe = x;
        let _pipe$1 = min(_pipe, max_bound);
        return max(_pipe$1, min_bound);
    }
}
function absolute_value(x) {
    let $ = x >= 0.0;
    if ($) return x;
    else return 0.0 - x;
}
function loosely_compare(a, b, tolerance) {
    let difference = absolute_value(a - b);
    let $ = difference <= tolerance;
    if ($) return new _orderMjs.Eq();
    else return compare(a, b);
}
function loosely_equals(a, b, tolerance) {
    let difference = absolute_value(a - b);
    return difference <= tolerance;
}
function power(base, exponent) {
    let fractional = (0, _gleamStdlibMjs.ceiling)(exponent) - exponent > 0.0;
    let $ = base < 0.0 && fractional || base === 0.0 && exponent < 0.0;
    if ($) return new (0, _gleamMjs.Error)(undefined);
    else return new (0, _gleamMjs.Ok)((0, _gleamStdlibMjs.power)(base, exponent));
}
function square_root(x) {
    return power(x, 0.5);
}
function negate(x) {
    return -1 * x;
}
function round(x) {
    let $ = x >= 0.0;
    if ($) return (0, _gleamStdlibMjs.round)(x);
    else return 0 - (0, _gleamStdlibMjs.round)(negate(x));
}
function to_precision(x, precision) {
    let $ = precision <= 0;
    if ($) {
        let factor = (0, _gleamStdlibMjs.power)(10.0, (0, _gleamStdlibMjs.identity)(-precision));
        return (0, _gleamStdlibMjs.identity)(round((0, _gleamMjs.divideFloat)(x, factor))) * factor;
    } else {
        let factor = (0, _gleamStdlibMjs.power)(10.0, (0, _gleamStdlibMjs.identity)(precision));
        return (0, _gleamMjs.divideFloat)((0, _gleamStdlibMjs.identity)(round(x * factor)), factor);
    }
}
function sum_loop(loop$numbers, loop$initial) {
    while(true){
        let numbers = loop$numbers;
        let initial = loop$initial;
        if (numbers instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first = numbers.head;
            let rest = numbers.tail;
            loop$numbers = rest;
            loop$initial = first + initial;
        }
    }
}
function sum(numbers) {
    return sum_loop(numbers, 0.0);
}
function product_loop(loop$numbers, loop$initial) {
    while(true){
        let numbers = loop$numbers;
        let initial = loop$initial;
        if (numbers instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first = numbers.head;
            let rest = numbers.tail;
            loop$numbers = rest;
            loop$initial = first * initial;
        }
    }
}
function product(numbers) {
    return product_loop(numbers, 1.0);
}
function modulo(dividend, divisor) {
    if (divisor === 0.0) return new (0, _gleamMjs.Error)(undefined);
    else return new (0, _gleamMjs.Ok)(dividend - (0, _gleamStdlibMjs.floor)((0, _gleamMjs.divideFloat)(dividend, divisor)) * divisor);
}
function divide(a, b) {
    if (b === 0.0) return new (0, _gleamMjs.Error)(undefined);
    else {
        let b$1 = b;
        return new (0, _gleamMjs.Ok)((0, _gleamMjs.divideFloat)(a, b$1));
    }
}
function add(a, b) {
    return a + b;
}
function multiply(a, b) {
    return a * b;
}
function subtract(a, b) {
    return a - b;
}
function logarithm(x) {
    let $ = x <= 0.0;
    if ($) return new (0, _gleamMjs.Error)(undefined);
    else return new (0, _gleamMjs.Ok)((0, _gleamStdlibMjs.log)(x));
}

},{"../gleam.mjs":"aiPrb","../gleam/order.mjs":"eYj92","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eYj92":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Lt", ()=>Lt);
parcelHelpers.export(exports, "Order$Lt", ()=>Order$Lt);
parcelHelpers.export(exports, "Order$isLt", ()=>Order$isLt);
parcelHelpers.export(exports, "Eq", ()=>Eq);
parcelHelpers.export(exports, "Order$Eq", ()=>Order$Eq);
parcelHelpers.export(exports, "Order$isEq", ()=>Order$isEq);
parcelHelpers.export(exports, "Gt", ()=>Gt);
parcelHelpers.export(exports, "Order$Gt", ()=>Order$Gt);
parcelHelpers.export(exports, "Order$isGt", ()=>Order$isGt);
/**
 * Inverts an order, so less-than becomes greater-than and greater-than
 * becomes less-than.
 *
 * ## Examples
 *
 * ```gleam
 * assert negate(Lt) == Gt
 * ```
 *
 * ```gleam
 * assert negate(Eq) == Eq
 * ```
 *
 * ```gleam
 * assert negate(Gt) == Lt
 * ```
 */ parcelHelpers.export(exports, "negate", ()=>negate);
/**
 * Produces a numeric representation of the order.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_int(Lt) == -1
 * ```
 *
 * ```gleam
 * assert to_int(Eq) == 0
 * ```
 *
 * ```gleam
 * assert to_int(Gt) == 1
 * ```
 */ parcelHelpers.export(exports, "to_int", ()=>to_int);
/**
 * Compares two `Order` values to one another, producing a new `Order`.
 *
 * ## Examples
 *
 * ```gleam
 * assert compare(Eq, with: Lt) == Gt
 * ```
 */ parcelHelpers.export(exports, "compare", ()=>compare);
/**
 * Inverts an ordering function, so less-than becomes greater-than and greater-than
 * becomes less-than.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 * import gleam/list
 *
 * assert list.sort([1, 5, 4], by: reverse(int.compare)) == [5, 4, 1]
 * ```
 */ parcelHelpers.export(exports, "reverse", ()=>reverse);
/**
 * Return a fallback `Order` in case the first argument is `Eq`.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 *
 * assert break_tie(in: int.compare(1, 1), with: Lt) == Lt
 * ```
 *
 * ```gleam
 * import gleam/int
 *
 * assert break_tie(in: int.compare(1, 0), with: Eq) == Gt
 * ```
 */ parcelHelpers.export(exports, "break_tie", ()=>break_tie);
/**
 * Invokes a fallback function returning an `Order` in case the first argument
 * is `Eq`.
 *
 * This can be useful when the fallback comparison might be expensive and it
 * needs to be delayed until strictly necessary.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 *
 * assert lazy_break_tie(in: int.compare(1, 1), with: fn() { Lt }) == Lt
 * ```
 *
 * ```gleam
 * import gleam/int
 *
 * assert lazy_break_tie(in: int.compare(1, 0), with: fn() { Eq }) == Gt
 * ```
 */ parcelHelpers.export(exports, "lazy_break_tie", ()=>lazy_break_tie);
var _gleamMjs = require("../gleam.mjs");
class Lt extends (0, _gleamMjs.CustomType) {
}
const Order$Lt = ()=>new Lt();
const Order$isLt = (value)=>value instanceof Lt;
class Eq extends (0, _gleamMjs.CustomType) {
}
const Order$Eq = ()=>new Eq();
const Order$isEq = (value)=>value instanceof Eq;
class Gt extends (0, _gleamMjs.CustomType) {
}
const Order$Gt = ()=>new Gt();
const Order$isGt = (value)=>value instanceof Gt;
function negate(order) {
    if (order instanceof Lt) return new Gt();
    else if (order instanceof Eq) return order;
    else return new Lt();
}
function to_int(order) {
    if (order instanceof Lt) return -1;
    else if (order instanceof Eq) return 0;
    else return 1;
}
function compare(a, b) {
    let x = a;
    let y = b;
    if ((0, _gleamMjs.isEqual)(x, y)) return new Eq();
    else if (a instanceof Lt) return new Lt();
    else if (a instanceof Eq && b instanceof Gt) return new Lt();
    else return new Gt();
}
function reverse(orderer) {
    return (a, b)=>{
        return orderer(b, a);
    };
}
function break_tie(order, other) {
    if (order instanceof Lt) return order;
    else if (order instanceof Eq) return other;
    else return order;
}
function lazy_break_tie(order, comparison) {
    if (order instanceof Lt) return order;
    else if (order instanceof Eq) return comparison();
    else return order;
}

},{"../gleam.mjs":"aiPrb","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"2eNPH":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "identity", ()=>identity);
parcelHelpers.export(exports, "parse_int", ()=>parse_int);
parcelHelpers.export(exports, "parse_float", ()=>parse_float);
parcelHelpers.export(exports, "to_string", ()=>to_string);
parcelHelpers.export(exports, "int_to_base_string", ()=>int_to_base_string);
parcelHelpers.export(exports, "int_from_base_string", ()=>int_from_base_string);
parcelHelpers.export(exports, "string_replace", ()=>string_replace);
parcelHelpers.export(exports, "string_reverse", ()=>string_reverse);
parcelHelpers.export(exports, "string_length", ()=>string_length);
parcelHelpers.export(exports, "graphemes", ()=>graphemes);
parcelHelpers.export(exports, "pop_grapheme", ()=>pop_grapheme);
parcelHelpers.export(exports, "pop_codeunit", ()=>pop_codeunit);
parcelHelpers.export(exports, "lowercase", ()=>lowercase);
parcelHelpers.export(exports, "uppercase", ()=>uppercase);
parcelHelpers.export(exports, "less_than", ()=>less_than);
parcelHelpers.export(exports, "add", ()=>add);
parcelHelpers.export(exports, "split", ()=>split);
parcelHelpers.export(exports, "concat", ()=>concat);
parcelHelpers.export(exports, "length", ()=>length);
parcelHelpers.export(exports, "string_byte_slice", ()=>string_byte_slice);
parcelHelpers.export(exports, "string_grapheme_slice", ()=>string_grapheme_slice);
parcelHelpers.export(exports, "string_codeunit_slice", ()=>string_codeunit_slice);
parcelHelpers.export(exports, "crop_string", ()=>crop_string);
parcelHelpers.export(exports, "contains_string", ()=>contains_string);
parcelHelpers.export(exports, "starts_with", ()=>starts_with);
parcelHelpers.export(exports, "ends_with", ()=>ends_with);
parcelHelpers.export(exports, "split_once", ()=>split_once);
parcelHelpers.export(exports, "trim_start", ()=>trim_start);
parcelHelpers.export(exports, "trim_end", ()=>trim_end);
parcelHelpers.export(exports, "bit_array_from_string", ()=>bit_array_from_string);
parcelHelpers.export(exports, "bit_array_bit_size", ()=>bit_array_bit_size);
parcelHelpers.export(exports, "bit_array_byte_size", ()=>bit_array_byte_size);
parcelHelpers.export(exports, "bit_array_pad_to_bytes", ()=>bit_array_pad_to_bytes);
parcelHelpers.export(exports, "bit_array_concat", ()=>bit_array_concat);
parcelHelpers.export(exports, "console_log", ()=>console_log);
parcelHelpers.export(exports, "console_error", ()=>console_error);
parcelHelpers.export(exports, "crash", ()=>crash);
parcelHelpers.export(exports, "bit_array_to_string", ()=>bit_array_to_string);
parcelHelpers.export(exports, "print", ()=>print);
parcelHelpers.export(exports, "print_error", ()=>print_error);
parcelHelpers.export(exports, "print_debug", ()=>print_debug);
parcelHelpers.export(exports, "ceiling", ()=>ceiling);
parcelHelpers.export(exports, "floor", ()=>floor);
parcelHelpers.export(exports, "round", ()=>round);
parcelHelpers.export(exports, "truncate", ()=>truncate);
parcelHelpers.export(exports, "power", ()=>power);
parcelHelpers.export(exports, "random_uniform", ()=>random_uniform);
parcelHelpers.export(exports, "bit_array_slice", ()=>bit_array_slice);
parcelHelpers.export(exports, "codepoint", ()=>codepoint);
parcelHelpers.export(exports, "string_to_codepoint_integer_list", ()=>string_to_codepoint_integer_list);
parcelHelpers.export(exports, "utf_codepoint_list_to_string", ()=>utf_codepoint_list_to_string);
parcelHelpers.export(exports, "utf_codepoint_to_int", ()=>utf_codepoint_to_int);
parcelHelpers.export(exports, "percent_decode", ()=>percent_decode);
parcelHelpers.export(exports, "percent_encode", ()=>percent_encode);
parcelHelpers.export(exports, "parse_query", ()=>parse_query);
// Implementation based on https://github.com/mitschabaude/fast-base64/blob/main/js.js
parcelHelpers.export(exports, "base64_encode", ()=>base64_encode);
// From https://developer.mozilla.org/en-US/docs/Glossary/Base64
parcelHelpers.export(exports, "base64_decode", ()=>base64_decode);
parcelHelpers.export(exports, "classify_dynamic", ()=>classify_dynamic);
parcelHelpers.export(exports, "byte_size", ()=>byte_size);
parcelHelpers.export(exports, "bitwise_and", ()=>bitwise_and);
parcelHelpers.export(exports, "bitwise_or", ()=>bitwise_or);
parcelHelpers.export(exports, "bitwise_exclusive_or", ()=>bitwise_exclusive_or);
parcelHelpers.export(exports, "bitwise_not", ()=>bitwise_not);
parcelHelpers.export(exports, "bitwise_shift_right", ()=>bitwise_shift_right);
parcelHelpers.export(exports, "bitwise_shift_left", ()=>bitwise_shift_left);
parcelHelpers.export(exports, "inspect", ()=>inspect);
parcelHelpers.export(exports, "float_to_string", ()=>float_to_string);
parcelHelpers.export(exports, "base16_encode", ()=>base16_encode);
parcelHelpers.export(exports, "base16_decode", ()=>base16_decode);
parcelHelpers.export(exports, "bit_array_to_int_and_size", ()=>bit_array_to_int_and_size);
parcelHelpers.export(exports, "bit_array_starts_with", ()=>bit_array_starts_with);
parcelHelpers.export(exports, "log", ()=>log);
parcelHelpers.export(exports, "exp", ()=>exp);
parcelHelpers.export(exports, "list_to_array", ()=>list_to_array);
parcelHelpers.export(exports, "index", ()=>index);
parcelHelpers.export(exports, "list", ()=>list);
parcelHelpers.export(exports, "dict", ()=>dict);
parcelHelpers.export(exports, "bit_array", ()=>bit_array);
parcelHelpers.export(exports, "float", ()=>float);
parcelHelpers.export(exports, "int", ()=>int);
parcelHelpers.export(exports, "string", ()=>string);
parcelHelpers.export(exports, "is_null", ()=>is_null);
var _gleamMjs = require("./gleam.mjs");
var _optionMjs = require("./gleam/option.mjs");
var _dictMjs = require("./dict.mjs");
var _dictMjsDefault = parcelHelpers.interopDefault(_dictMjs);
var _dynamicMjs = require("./gleam/dynamic.mjs");
var _decodeMjs = require("./gleam/dynamic/decode.mjs");
const Nil = undefined;
function identity(x) {
    return x;
}
function parse_int(value) {
    if (/^[-+]?(\d+)$/.test(value)) return (0, _gleamMjs.Result$Ok)(parseInt(value));
    else return (0, _gleamMjs.Result$Error)(Nil);
}
function parse_float(value) {
    if (/^[-+]?(\d+)\.(\d+)([eE][-+]?\d+)?$/.test(value)) return (0, _gleamMjs.Result$Ok)(parseFloat(value));
    else return (0, _gleamMjs.Result$Error)(Nil);
}
function to_string(term) {
    return term.toString();
}
function int_to_base_string(int, base) {
    return int.toString(base).toUpperCase();
}
const int_base_patterns = {
    2: /[^0-1]/,
    3: /[^0-2]/,
    4: /[^0-3]/,
    5: /[^0-4]/,
    6: /[^0-5]/,
    7: /[^0-6]/,
    8: /[^0-7]/,
    9: /[^0-8]/,
    10: /[^0-9]/,
    11: /[^0-9a]/,
    12: /[^0-9a-b]/,
    13: /[^0-9a-c]/,
    14: /[^0-9a-d]/,
    15: /[^0-9a-e]/,
    16: /[^0-9a-f]/,
    17: /[^0-9a-g]/,
    18: /[^0-9a-h]/,
    19: /[^0-9a-i]/,
    20: /[^0-9a-j]/,
    21: /[^0-9a-k]/,
    22: /[^0-9a-l]/,
    23: /[^0-9a-m]/,
    24: /[^0-9a-n]/,
    25: /[^0-9a-o]/,
    26: /[^0-9a-p]/,
    27: /[^0-9a-q]/,
    28: /[^0-9a-r]/,
    29: /[^0-9a-s]/,
    30: /[^0-9a-t]/,
    31: /[^0-9a-u]/,
    32: /[^0-9a-v]/,
    33: /[^0-9a-w]/,
    34: /[^0-9a-x]/,
    35: /[^0-9a-y]/,
    36: /[^0-9a-z]/
};
function int_from_base_string(string, base) {
    if (int_base_patterns[base].test(string.replace(/^-/, "").toLowerCase())) return (0, _gleamMjs.Result$Error)(Nil);
    const result = parseInt(string, base);
    if (isNaN(result)) return (0, _gleamMjs.Result$Error)(Nil);
    return (0, _gleamMjs.Result$Ok)(result);
}
function string_replace(string, target, substitute) {
    return string.replaceAll(target, substitute);
}
function string_reverse(string) {
    return [
        ...string
    ].reverse().join("");
}
function string_length(string) {
    if (string === "") return 0;
    const iterator = graphemes_iterator(string);
    if (iterator) {
        let i = 0;
        for (const _ of iterator)i++;
        return i;
    } else return string.match(/./gsu).length;
}
function graphemes(string) {
    const iterator = graphemes_iterator(string);
    if (iterator) return arrayToList(Array.from(iterator).map((item)=>item.segment));
    else return arrayToList(string.match(/./gsu));
}
let segmenter = undefined;
function graphemes_iterator(string) {
    if (globalThis.Intl && Intl.Segmenter) {
        segmenter ||= new Intl.Segmenter();
        return segmenter.segment(string)[Symbol.iterator]();
    }
}
function pop_grapheme(string) {
    let first;
    const iterator = graphemes_iterator(string);
    if (iterator) first = iterator.next().value?.segment;
    else first = string.match(/./su)?.[0];
    if (first) return (0, _gleamMjs.Result$Ok)([
        first,
        string.slice(first.length)
    ]);
    else return (0, _gleamMjs.Result$Error)(Nil);
}
function pop_codeunit(str) {
    return [
        str.charCodeAt(0) | 0,
        str.slice(1)
    ];
}
function lowercase(string) {
    return string.toLowerCase();
}
function uppercase(string) {
    return string.toUpperCase();
}
function less_than(a, b) {
    return a < b;
}
function add(a, b) {
    return a + b;
}
function split(xs, pattern) {
    return arrayToList(xs.split(pattern));
}
function concat(xs) {
    let result = "";
    for (const x of xs)result = result + x;
    return result;
}
function length(data) {
    return data.length;
}
function string_byte_slice(string, index, length) {
    return string.slice(index, index + length);
}
function string_grapheme_slice(string, idx, len) {
    if (len <= 0 || idx >= string.length) return "";
    const iterator = graphemes_iterator(string);
    if (iterator) {
        while(idx-- > 0)iterator.next();
        let result = "";
        while(len-- > 0){
            const v = iterator.next().value;
            if (v === undefined) break;
            result += v.segment;
        }
        return result;
    } else return string.match(/./gsu).slice(idx, idx + len).join("");
}
function string_codeunit_slice(str, from, length) {
    return str.slice(from, from + length);
}
function crop_string(string, substring) {
    return string.substring(string.indexOf(substring));
}
function contains_string(haystack, needle) {
    return haystack.indexOf(needle) >= 0;
}
function starts_with(haystack, needle) {
    return haystack.startsWith(needle);
}
function ends_with(haystack, needle) {
    return haystack.endsWith(needle);
}
function split_once(haystack, needle) {
    const index = haystack.indexOf(needle);
    if (index >= 0) {
        const before = haystack.slice(0, index);
        const after = haystack.slice(index + needle.length);
        return (0, _gleamMjs.Result$Ok)([
            before,
            after
        ]);
    } else return (0, _gleamMjs.Result$Error)(Nil);
}
const unicode_whitespaces = [
    "\u0020",
    "\u0009",
    "\u000A",
    "\u000B",
    "\u000C",
    "\u000D",
    "\u0085",
    "\u2028",
    "\u2029"
].join("");
const trim_start_regex = /* @__PURE__ */ new RegExp(`^[${unicode_whitespaces}]*`);
const trim_end_regex = /* @__PURE__ */ new RegExp(`[${unicode_whitespaces}]*$`);
function trim_start(string) {
    return string.replace(trim_start_regex, "");
}
function trim_end(string) {
    return string.replace(trim_end_regex, "");
}
function bit_array_from_string(string) {
    return (0, _gleamMjs.toBitArray)([
        (0, _gleamMjs.stringBits)(string)
    ]);
}
function bit_array_bit_size(bit_array) {
    return bit_array.bitSize;
}
function bit_array_byte_size(bit_array) {
    return bit_array.byteSize;
}
function bit_array_pad_to_bytes(bit_array) {
    const trailingBitsCount = bit_array.bitSize % 8;
    // If the bit array is a whole number of bytes it can be returned unchanged
    if (trailingBitsCount === 0) return bit_array;
    const finalByte = bit_array.byteAt(bit_array.byteSize - 1);
    // The required final byte has its unused trailing bits set to zero
    const unusedBitsCount = 8 - trailingBitsCount;
    const correctFinalByte = finalByte >> unusedBitsCount << unusedBitsCount;
    // If the unused bits in the final byte are already set to zero then the
    // existing buffer can be re-used, avoiding a copy
    if (finalByte === correctFinalByte) return new (0, _gleamMjs.BitArray)(bit_array.rawBuffer, bit_array.byteSize * 8, bit_array.bitOffset);
    // Copy the bit array into a new aligned buffer and set the correct final byte
    const buffer = new Uint8Array(bit_array.byteSize);
    for(let i = 0; i < buffer.length - 1; i++)buffer[i] = bit_array.byteAt(i);
    buffer[buffer.length - 1] = correctFinalByte;
    return new (0, _gleamMjs.BitArray)(buffer);
}
function bit_array_concat(bit_arrays) {
    return (0, _gleamMjs.toBitArray)(bit_arrays.toArray());
}
function console_log(term) {
    console.log(term);
}
function console_error(term) {
    console.error(term);
}
function crash(message) {
    throw new globalThis.Error(message);
}
function bit_array_to_string(bit_array) {
    // If the bit array isn't a whole number of bytes then return an error
    if (bit_array.bitSize % 8 !== 0) return (0, _gleamMjs.Result$Error)(Nil);
    try {
        const decoder = new TextDecoder("utf-8", {
            fatal: true
        });
        if (bit_array.bitOffset === 0) return (0, _gleamMjs.Result$Ok)(decoder.decode(bit_array.rawBuffer));
        else {
            // The input data isn't aligned, so copy it into a new aligned buffer so
            // that TextDecoder can be used
            const buffer = new Uint8Array(bit_array.byteSize);
            for(let i = 0; i < buffer.length; i++)buffer[i] = bit_array.byteAt(i);
            return (0, _gleamMjs.Result$Ok)(decoder.decode(buffer));
        }
    } catch  {
        return (0, _gleamMjs.Result$Error)(Nil);
    }
}
function print(string) {
    if (typeof Deno === "object") Deno.stdout.writeSync(new TextEncoder().encode(string)); // We can write without a trailing newline
    else console.log(string); // We're in a browser. Newlines are mandated
}
function print_error(string) {
    if (typeof Deno === "object") Deno.stderr.writeSync(new TextEncoder().encode(string)); // We can write without a trailing newline
    else console.error(string); // We're in a browser. Newlines are mandated
}
function print_debug(string) {
    if (typeof Deno === "object") Deno.stderr.writeSync(new TextEncoder().encode(string + "\n")); // If we're in Deno, use `stderr`
    else console.log(string); // Otherwise, use `console.log` (so that it doesn't look like an error)
}
function ceiling(float) {
    return Math.ceil(float);
}
function floor(float) {
    return Math.floor(float);
}
function round(float) {
    return Math.round(float);
}
function truncate(float) {
    return Math.trunc(float);
}
function power(base, exponent) {
    // It is checked in Gleam that:
    // - The base is non-negative and that the exponent is not fractional.
    // - The base is non-zero and the exponent is non-negative (otherwise
    //   the result will essentially be division by zero).
    // It can thus be assumed that valid input is passed to the Math.pow
    // function and a NaN or Infinity value will not be produced.
    return Math.pow(base, exponent);
}
function random_uniform() {
    const random_uniform_result = Math.random();
    // With round-to-nearest-even behavior, the ranges claimed for the functions below
    // (excluding the one for Math.random() itself) aren't exact.
    // If extremely large bounds are chosen (2^53 or higher),
    // it's possible in extremely rare cases to calculate the usually-excluded upper bound.
    // Note that as numbers in JavaScript are IEEE 754 floating point numbers
    // See: <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random>
    // Because of this, we just loop 'until' we get a valid result where 0.0 <= x < 1.0:
    if (random_uniform_result === 1.0) return random_uniform();
    return random_uniform_result;
}
function bit_array_slice(bits, position, length) {
    const start = Math.min(position, position + length);
    const end = Math.max(position, position + length);
    if (start < 0 || end * 8 > bits.bitSize) return (0, _gleamMjs.Result$Error)(Nil);
    return (0, _gleamMjs.Result$Ok)((0, _gleamMjs.bitArraySlice)(bits, start * 8, end * 8));
}
function codepoint(int) {
    return new (0, _gleamMjs.UtfCodepoint)(int);
}
function string_to_codepoint_integer_list(string) {
    return arrayToList(Array.from(string).map((item)=>item.codePointAt(0)));
}
function utf_codepoint_list_to_string(utf_codepoint_integer_list) {
    return utf_codepoint_integer_list.toArray().map((x)=>String.fromCodePoint(x.value)).join("");
}
function utf_codepoint_to_int(utf_codepoint) {
    return utf_codepoint.value;
}
function unsafe_percent_decode(string) {
    return decodeURIComponent(string || "");
}
function unsafe_percent_decode_query(string) {
    return decodeURIComponent((string || "").replace("+", " "));
}
function percent_decode(string) {
    try {
        return (0, _gleamMjs.Result$Ok)(unsafe_percent_decode(string));
    } catch  {
        return (0, _gleamMjs.Result$Error)(Nil);
    }
}
function percent_encode(string) {
    return encodeURIComponent(string).replace("%2B", "+");
}
function parse_query(query) {
    try {
        const pairs = [];
        for (const section of query.split("&")){
            const [key, value] = section.split("=");
            if (!key) continue;
            const decodedKey = unsafe_percent_decode_query(key);
            const decodedValue = unsafe_percent_decode_query(value);
            pairs.push([
                decodedKey,
                decodedValue
            ]);
        }
        return (0, _gleamMjs.Result$Ok)(arrayToList(pairs));
    } catch  {
        return (0, _gleamMjs.Result$Error)(Nil);
    }
}
const b64EncodeLookup = [
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    43,
    47
];
let b64TextDecoder;
function base64_encode(bit_array, padding) {
    b64TextDecoder ??= new TextDecoder();
    bit_array = bit_array_pad_to_bytes(bit_array);
    const m = bit_array.byteSize;
    const k = m % 3;
    const n = Math.floor(m / 3) * 4 + (k && k + 1);
    const N = Math.ceil(m / 3) * 4;
    const encoded = new Uint8Array(N);
    for(let i = 0, j = 0; j < m; i += 4, j += 3){
        const y = (bit_array.byteAt(j) << 16) + (bit_array.byteAt(j + 1) << 8) + (bit_array.byteAt(j + 2) | 0);
        encoded[i] = b64EncodeLookup[y >> 18];
        encoded[i + 1] = b64EncodeLookup[y >> 12 & 0x3f];
        encoded[i + 2] = b64EncodeLookup[y >> 6 & 0x3f];
        encoded[i + 3] = b64EncodeLookup[y & 0x3f];
    }
    let base64 = b64TextDecoder.decode(new Uint8Array(encoded.buffer, 0, n));
    if (padding) {
        if (k === 1) base64 += "==";
        else if (k === 2) base64 += "=";
    }
    return base64;
}
function base64_decode(sBase64) {
    try {
        const binString = atob(sBase64);
        const length = binString.length;
        const array = new Uint8Array(length);
        for(let i = 0; i < length; i++)array[i] = binString.charCodeAt(i);
        return (0, _gleamMjs.Result$Ok)(new (0, _gleamMjs.BitArray)(array));
    } catch  {
        return (0, _gleamMjs.Result$Error)(Nil);
    }
}
function classify_dynamic(data) {
    if (typeof data === "string") return "String";
    else if (typeof data === "boolean") return "Bool";
    else if (isResult(data)) return "Result";
    else if (isList(data)) return "List";
    else if (data instanceof (0, _gleamMjs.BitArray)) return "BitArray";
    else if (data instanceof (0, _dictMjsDefault.default)) return "Dict";
    else if (Number.isInteger(data)) return "Int";
    else if (Array.isArray(data)) return `Array`;
    else if (typeof data === "number") return "Float";
    else if (data === null) return "Nil";
    else if (data === undefined) return "Nil";
    else {
        const type = typeof data;
        return type.charAt(0).toUpperCase() + type.slice(1);
    }
}
function byte_size(string) {
    return new TextEncoder().encode(string).length;
}
// In JavaScript, bitwise operations convert numbers to a sequence of 32 bits,
// while Erlang uses arbitrary precision integers.
//
// To get around this, every function here follows this pattern:
//
// 1. If both values fit in the 32-bit signed integer range, use the standard
//    JavaScript bitwise operators directly.
//
//    Note: For bitwise_shift_left, the result also needs to fit in 32 bits,
//    so we use floating-point multiplication instead.
//
// 2. If either value falls outside the safe integer range (-2^53, 2^53),
//    fall back to BigInt arithmetic, then downcast the result back to a Number.
//
// 3. Otherwise (safe integers outside the 32-bit range), we split the operation
//    across the high 21 bits and low 32 bits individually:
//
//        x1 $ x2 = ((hi(x1) $ hi(x2)) << 32) | (lo(x1) $ lo(x2))
//
//    where `$` is a bitwise operator.
//
//    We split both values into a `hi` and a `lo` part:
//
//        hi(x) = Math.floor(x / 2^32)    the upper 21 bits
//        lo(x) = x >>> 0                 the lower 32 bits (as unsigned)
//
//    For `hi`, we use that shifts are equal to multiplication/division with
//    powers of two to get around the 32-bit range limitation. Math.floor is
//    used instead of truncation since arithmetic right shift fills with the
//    sign bit. For negative numbers, the discarded bits were non-zero
//    (representing a positive fractional part), so discarding them makes the
//    result strictly more negative, i.e. rounding away from 0.
//
//    This works because bitwise operators are distributive across bit ranges:
//
//        x1 $ x2 = (hi(x1) $ hi(x2)) << 32 | (lo(x1) $ lo(x2))
//                = (hi(x1) $ hi(x2)) * 2^32 + (lo(x1) $ lo(x2))
//
//    JavaScript bitwise operators truncate inputs to signed 32-bit integers,
//    so `x1 $ x2` already computes `lo(x1) $ lo(x2)` — we just need to
//    reinterpret the signed result as unsigned using `>>> 0`:
//
//        lo(x1) $ lo(x2) = (x1 $ x2) >>> 0 = lo(x1 $ x2)
//        => x1 $ x2 = (hi(x1) $ hi(x2)) * 2^32 + lo(x1 $ x2).
//
const MIN_I32 = -(2 ** 31); // -2147483648
const MAX_I32 = 2 ** 31 - 1; //  2147483647
const U32 = 2 ** 32;
const MAX_SAFE = Number.MAX_SAFE_INTEGER;
const MIN_SAFE = Number.MIN_SAFE_INTEGER;
function bitwise_and(x, y) {
    if (x >= MIN_I32 && x <= MAX_I32 && y >= MIN_I32 && y <= MAX_I32) return x & y;
    if (x < MIN_SAFE || x > MAX_SAFE || y < MIN_SAFE || y > MAX_SAFE) return Number(BigInt(x) & BigInt(y));
    return (Math.floor(x / U32) & Math.floor(y / U32)) * U32 + ((x & y) >>> 0);
}
function bitwise_or(x, y) {
    if (x >= MIN_I32 && x <= MAX_I32 && y >= MIN_I32 && y <= MAX_I32) return x | y;
    if (x < MIN_SAFE || x > MAX_SAFE || y < MIN_SAFE || y > MAX_SAFE) return Number(BigInt(x) | BigInt(y));
    return (Math.floor(x / U32) | Math.floor(y / U32)) * U32 + ((x | y) >>> 0);
}
function bitwise_exclusive_or(x, y) {
    if (x >= MIN_I32 && x <= MAX_I32 && y >= MIN_I32 && y <= MAX_I32) return x ^ y;
    if (x < MIN_SAFE || x > MAX_SAFE || y < MIN_SAFE || y > MAX_SAFE) return Number(BigInt(x) ^ BigInt(y));
    return (Math.floor(x / U32) ^ Math.floor(y / U32)) * U32 + ((x ^ y) >>> 0);
}
function bitwise_not(x) {
    if (x >= MIN_I32 && x <= MAX_I32) return ~x;
    if (x < MIN_SAFE || x > MAX_SAFE) return Number(~BigInt(x));
    return ~Math.floor(x / U32) * U32 + (~x >>> 0);
}
function bitwise_shift_right(x, y) {
    if (y === 0) return x;
    if (y < 0) return bitwise_shift_left(x, -y);
    if (y < 32 && x >= MIN_I32 && x <= MAX_I32) return x >> y;
    if (x < MIN_SAFE || x > MAX_SAFE) return Number(BigInt(x) >> BigInt(y));
    const ahi = Math.floor(x / U32);
    // Shifting right by y < 32 moves bits across the hi/lo boundary:
    //
    //   before: [ hi (21 bits) | lo (32 bits) ]
    //   after:  [ hi >> y      | (hi's low y bits) ++ (lo >> y) ]
    //
    // The new low word has two sources:
    //   - lo's bits shifted down:        x >>> y   (>>> treats x as unsigned 32-bit)
    //   - hi's bottom y bits shifted up: ahi << (32 - y).
    if (y < 32) return (ahi >> y) * U32 + ((x >>> y | ahi << 32 - y) >>> 0);
    // Shifting by >= 32 wipes out the entire low word. The result is just the
    // high word shifted right by the remaining amount.
    return ahi >> y - 32;
}
function bitwise_shift_left(x, y) {
    if (y === 0) return x;
    if (y < 0) return bitwise_shift_right(x, -y);
    if (y < 31) return x * (1 << y);
    return x * 2 ** y;
}
function inspect(v) {
    return new Inspector().inspect(v);
}
function float_to_string(float) {
    const string = float.toString().replace("+", "");
    if (string.indexOf(".") >= 0) return string;
    else {
        const index = string.indexOf("e");
        if (index >= 0) return string.slice(0, index) + ".0" + string.slice(index);
        else return string + ".0";
    }
}
class Inspector {
    #references = new Set();
    inspect(v) {
        const t = typeof v;
        if (v === true) return "True";
        if (v === false) return "False";
        if (v === null) return "//js(null)";
        if (v === undefined) return "Nil";
        if (t === "string") return this.#string(v);
        if (t === "bigint" || Number.isInteger(v)) return v.toString();
        if (t === "number") return float_to_string(v);
        if (v instanceof (0, _gleamMjs.UtfCodepoint)) return this.#utfCodepoint(v);
        if (v instanceof (0, _gleamMjs.BitArray)) return this.#bit_array(v);
        if (v instanceof RegExp) return `//js(${v})`;
        if (v instanceof Date) return `//js(Date("${v.toISOString()}"))`;
        if (v instanceof globalThis.Error) return `//js(${v.toString()})`;
        if (v instanceof Function) {
            const args = [];
            for (const i of Array(v.length).keys())args.push(String.fromCharCode(i + 97));
            return `//fn(${args.join(", ")}) { ... }`;
        }
        if (this.#references.size === this.#references.add(v).size) return "//js(circular reference)";
        let printed;
        if (Array.isArray(v)) printed = `#(${v.map((v)=>this.inspect(v)).join(", ")})`;
        else if (isList(v)) printed = this.#list(v);
        else if (v instanceof (0, _gleamMjs.CustomType)) printed = this.#customType(v);
        else if (v instanceof (0, _dictMjsDefault.default)) printed = this.#dict(v);
        else if (v instanceof Set) return `//js(Set(${[
            ...v
        ].map((v)=>this.inspect(v)).join(", ")}))`;
        else printed = this.#object(v);
        this.#references.delete(v);
        return printed;
    }
    #object(v) {
        const name = Object.getPrototypeOf(v)?.constructor?.name || "Object";
        const props = [];
        for (const k of Object.keys(v))props.push(`${this.inspect(k)}: ${this.inspect(v[k])}`);
        const body = props.length ? " " + props.join(", ") + " " : "";
        const head = name === "Object" ? "" : name + " ";
        return `//js(${head}{${body}})`;
    }
    #dict(map) {
        let body = "dict.from_list([";
        let first = true;
        body = (0, _dictMjs.fold)(map, body, (body, key, value)=>{
            if (!first) body = body + ", ";
            first = false;
            return body + "#(" + this.inspect(key) + ", " + this.inspect(value) + ")";
        });
        return body + "])";
    }
    #customType(record) {
        const props = Object.keys(record).map((label)=>{
            const value = this.inspect(record[label]);
            return isNaN(parseInt(label)) ? `${label}: ${value}` : value;
        }).join(", ");
        return props ? `${record.constructor.name}(${props})` : record.constructor.name;
    }
    #list(list) {
        if ((0, _gleamMjs.List$isEmpty)(list)) return "[]";
        let char_out = 'charlist.from_string("';
        let list_out = "[";
        let current = list;
        while((0, _gleamMjs.List$isNonEmpty)(current)){
            let element = current.head;
            current = current.tail;
            if (list_out !== "[") list_out += ", ";
            list_out += this.inspect(element);
            if (char_out) {
                if (Number.isInteger(element) && element >= 32 && element <= 126) char_out += String.fromCharCode(element);
                else char_out = null;
            }
        }
        if (char_out) return char_out + '")';
        else return list_out + "]";
    }
    #string(str) {
        let new_str = '"';
        for(let i = 0; i < str.length; i++){
            const char = str[i];
            switch(char){
                case "\n":
                    new_str += "\\n";
                    break;
                case "\r":
                    new_str += "\\r";
                    break;
                case "\t":
                    new_str += "\\t";
                    break;
                case "\f":
                    new_str += "\\f";
                    break;
                case "\\":
                    new_str += "\\\\";
                    break;
                case '"':
                    new_str += '\\"';
                    break;
                default:
                    if (char < " " || char > "~" && char < "\xa0") new_str += "\\u{" + char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0") + "}";
                    else new_str += char;
            }
        }
        new_str += '"';
        return new_str;
    }
    #utfCodepoint(codepoint) {
        return `//utfcodepoint(${String.fromCodePoint(codepoint.value)})`;
    }
    #bit_array(bits) {
        if (bits.bitSize === 0) return "<<>>";
        let acc = "<<";
        for(let i = 0; i < bits.byteSize - 1; i++){
            acc += bits.byteAt(i).toString();
            acc += ", ";
        }
        if (bits.byteSize * 8 === bits.bitSize) acc += bits.byteAt(bits.byteSize - 1).toString();
        else {
            const trailingBitsCount = bits.bitSize % 8;
            acc += bits.byteAt(bits.byteSize - 1) >> 8 - trailingBitsCount;
            acc += `:size(${trailingBitsCount})`;
        }
        acc += ">>";
        return acc;
    }
}
function base16_encode(bit_array) {
    const trailingBitsCount = bit_array.bitSize % 8;
    let result = "";
    for(let i = 0; i < bit_array.byteSize; i++){
        let byte = bit_array.byteAt(i);
        if (i === bit_array.byteSize - 1 && trailingBitsCount !== 0) {
            const unusedBitsCount = 8 - trailingBitsCount;
            byte = byte >> unusedBitsCount << unusedBitsCount;
        }
        result += byte.toString(16).padStart(2, "0").toUpperCase();
    }
    return result;
}
function base16_decode(string) {
    const bytes = new Uint8Array(string.length / 2);
    for(let i = 0; i < string.length; i += 2){
        const a = parseInt(string[i], 16);
        const b = parseInt(string[i + 1], 16);
        if (isNaN(a) || isNaN(b)) return (0, _gleamMjs.Result$Error)(Nil);
        bytes[i / 2] = a * 16 + b;
    }
    return (0, _gleamMjs.Result$Ok)(new (0, _gleamMjs.BitArray)(bytes));
}
function bit_array_to_int_and_size(bits) {
    const trailingBitsCount = bits.bitSize % 8;
    const unusedBitsCount = trailingBitsCount === 0 ? 0 : 8 - trailingBitsCount;
    return [
        bits.byteAt(0) >> unusedBitsCount,
        bits.bitSize
    ];
}
function bit_array_starts_with(bits, prefix) {
    if (prefix.bitSize > bits.bitSize) return false;
    // Check any whole bytes
    const byteCount = Math.trunc(prefix.bitSize / 8);
    for(let i = 0; i < byteCount; i++){
        if (bits.byteAt(i) !== prefix.byteAt(i)) return false;
    }
    // Check any trailing bits at the end of the prefix
    if (prefix.bitSize % 8 !== 0) {
        const unusedBitsCount = 8 - prefix.bitSize % 8;
        if (bits.byteAt(byteCount) >> unusedBitsCount !== prefix.byteAt(byteCount) >> unusedBitsCount) return false;
    }
    return true;
}
function log(x) {
    // It is checked in Gleam that:
    // - The input is strictly positive (x > 0)
    // - This ensures that Math.log will never return NaN or -Infinity
    // The function can thus safely pass the input to Math.log
    // and a valid finite float will always be produced.
    return Math.log(x);
}
function exp(x) {
    return Math.exp(x);
}
function list_to_array(list) {
    let current = list;
    let array = [];
    while((0, _gleamMjs.List$isNonEmpty)(current)){
        array.push(current.head);
        current = current.tail;
    }
    return array;
}
function index(data, key) {
    // Dictionaries and dictionary-like objects can be indexed
    if (data instanceof (0, _dictMjsDefault.default)) {
        const result = (0, _dictMjs.get)(data, key);
        return (0, _gleamMjs.Result$Ok)(result.isOk() ? new (0, _optionMjs.Some)(result[0]) : new (0, _optionMjs.None)());
    }
    if (data instanceof WeakMap || data instanceof Map) {
        const token = {};
        const entry = data.get(key, token);
        if (entry === token) return (0, _gleamMjs.Result$Ok)(new (0, _optionMjs.None)());
        return (0, _gleamMjs.Result$Ok)(new (0, _optionMjs.Some)(entry));
    }
    const key_is_int = Number.isInteger(key);
    // Only elements 0-7 of lists can be indexed, negative indices are not allowed
    if (key_is_int && key >= 0 && key < 8 && isList(data)) {
        let i = 0;
        for (const value of data){
            if (i === key) return (0, _gleamMjs.Result$Ok)(new (0, _optionMjs.Some)(value));
            i++;
        }
        return (0, _gleamMjs.Result$Error)("Indexable");
    }
    // Arrays and objects can be indexed
    if (key_is_int && Array.isArray(data) || data && typeof data === "object" || data && Object.getPrototypeOf(data) === Object.prototype) {
        if (key in data) return (0, _gleamMjs.Result$Ok)(new (0, _optionMjs.Some)(data[key]));
        return (0, _gleamMjs.Result$Ok)(new (0, _optionMjs.None)());
    }
    return (0, _gleamMjs.Result$Error)(key_is_int ? "Indexable" : "Dict");
}
function list(data, decode, pushPath, index, emptyList) {
    if (!(isList(data) || Array.isArray(data))) {
        const error = (0, _decodeMjs.DecodeError$DecodeError)("List", (0, _dynamicMjs.classify)(data), emptyList);
        return [
            emptyList,
            arrayToList([
                error
            ])
        ];
    }
    const decoded = [];
    for (const element of data){
        const layer = decode(element);
        const [out, errors] = layer;
        if ((0, _gleamMjs.List$isNonEmpty)(errors)) {
            const [_, errors] = pushPath(layer, index.toString());
            return [
                emptyList,
                errors
            ];
        }
        decoded.push(out);
        index++;
    }
    return [
        arrayToList(decoded),
        emptyList
    ];
}
function dict(data) {
    if (data instanceof (0, _dictMjsDefault.default)) return (0, _gleamMjs.Result$Ok)(data);
    if (data instanceof Map || data instanceof WeakMap) return (0, _gleamMjs.Result$Ok)((0, _dictMjs.from)(data));
    if (data == null) return (0, _gleamMjs.Result$Error)("Dict");
    if (typeof data !== "object") return (0, _gleamMjs.Result$Error)("Dict");
    const proto = Object.getPrototypeOf(data);
    if (proto === Object.prototype || proto === null) return (0, _gleamMjs.Result$Ok)((0, _dictMjs.from)(Object.entries(data)));
    return (0, _gleamMjs.Result$Error)("Dict");
}
function bit_array(data) {
    if (data instanceof (0, _gleamMjs.BitArray)) return (0, _gleamMjs.Result$Ok)(data);
    if (data instanceof Uint8Array) return (0, _gleamMjs.Result$Ok)(new (0, _gleamMjs.BitArray)(data));
    return (0, _gleamMjs.Result$Error)(new (0, _gleamMjs.BitArray)(new Uint8Array()));
}
function float(data) {
    if (typeof data === "number") return (0, _gleamMjs.Result$Ok)(data);
    return (0, _gleamMjs.Result$Error)(0.0);
}
function int(data) {
    if (Number.isInteger(data)) return (0, _gleamMjs.Result$Ok)(data);
    return (0, _gleamMjs.Result$Error)(0);
}
function string(data) {
    if (typeof data === "string") return (0, _gleamMjs.Result$Ok)(data);
    return (0, _gleamMjs.Result$Error)("");
}
function is_null(data) {
    return data === null || data === undefined;
}
function arrayToList(array) {
    let list = (0, _gleamMjs.List$Empty)();
    let i = array.length;
    while(i--)list = (0, _gleamMjs.List$NonEmpty)(array[i], list);
    return list;
}
function isList(data) {
    return (0, _gleamMjs.List$isEmpty)(data) || (0, _gleamMjs.List$isNonEmpty)(data);
}
function isResult(data) {
    return (0, _gleamMjs.Result$isOk)(data) || (0, _gleamMjs.Result$isError)(data);
}

},{"./gleam.mjs":"aiPrb","./gleam/option.mjs":"aWtoH","./dict.mjs":"287yP","./gleam/dynamic.mjs":"iAWCk","./gleam/dynamic/decode.mjs":"gmHd7","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"iAWCk":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "array", ()=>(0, _gleamStdlibMjs.list_to_array));
parcelHelpers.export(exports, "bit_array", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "bool", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "classify", ()=>(0, _gleamStdlibMjs.classify_dynamic));
parcelHelpers.export(exports, "float", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "int", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "list", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "string", ()=>(0, _gleamStdlibMjs.identity));
/**
 * Create a dynamic value made of an unordered series of keys and values, where
 * the keys are unique.
 *
 * On Erlang this will be a map, on JavaScript this will be a Gleam dict
 * object.
 */ parcelHelpers.export(exports, "properties", ()=>properties);
/**
 * A dynamic value representing nothing.
 *
 * On Erlang this will be the atom `nil`, on JavaScript this will be
 * `undefined`.
 */ parcelHelpers.export(exports, "nil", ()=>nil);
var _dictMjs = require("../gleam/dict.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
function properties(entries) {
    return (0, _gleamStdlibMjs.identity)(_dictMjs.from_list(entries));
}
function nil() {
    return (0, _gleamStdlibMjs.identity)(undefined);
}

},{"../gleam/dict.mjs":"b8yrU","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"gmHd7":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "DecodeError", ()=>DecodeError);
parcelHelpers.export(exports, "DecodeError$DecodeError", ()=>DecodeError$DecodeError);
parcelHelpers.export(exports, "DecodeError$isDecodeError", ()=>DecodeError$isDecodeError);
parcelHelpers.export(exports, "DecodeError$DecodeError$expected", ()=>DecodeError$DecodeError$expected);
parcelHelpers.export(exports, "DecodeError$DecodeError$0", ()=>DecodeError$DecodeError$0);
parcelHelpers.export(exports, "DecodeError$DecodeError$found", ()=>DecodeError$DecodeError$found);
parcelHelpers.export(exports, "DecodeError$DecodeError$1", ()=>DecodeError$DecodeError$1);
parcelHelpers.export(exports, "DecodeError$DecodeError$path", ()=>DecodeError$DecodeError$path);
parcelHelpers.export(exports, "DecodeError$DecodeError$2", ()=>DecodeError$DecodeError$2);
parcelHelpers.export(exports, "dynamic", ()=>dynamic);
parcelHelpers.export(exports, "bool", ()=>bool);
parcelHelpers.export(exports, "int", ()=>int);
parcelHelpers.export(exports, "float", ()=>float);
parcelHelpers.export(exports, "bit_array", ()=>bit_array);
parcelHelpers.export(exports, "string", ()=>string);
/**
 * Run a decoder on a `Dynamic` value, decoding the value if it is of the
 * desired type, or returning errors.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = {
 *   use name <- decode.field("name", decode.string)
 *   use email <- decode.field("email", decode.string)
 *   decode.success(SignUp(name: name, email: email))
 * }
 *
 * decode.run(data, decoder)
 * ```
 */ parcelHelpers.export(exports, "run", ()=>run);
/**
 * Finalise a decoder having successfully extracted a value.
 *
 * ## Examples
 *
 * ```gleam
 * let data = dynamic.properties([
 *   #(dynamic.string("email"), dynamic.string("lucy@example.com")),
 *   #(dynamic.string("name"), dynamic.string("Lucy")),
 * ])
 *
 * let decoder = {
 *   use name <- decode.field("name", string)
 *   use email <- decode.field("email", string)
 *   decode.success(SignUp(name: name, email: email))
 * }
 *
 * let result = decode.run(data, decoder)
 * assert result == Ok(SignUp(name: "Lucy", email: "lucy@example.com"))
 * ```
 */ parcelHelpers.export(exports, "success", ()=>success);
/**
 * Apply a transformation function to any value decoded by the decoder.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = decode.int |> decode.map(int.to_string)
 * let result = decode.run(dynamic.int(1000), decoder)
 * assert result == Ok("1000")
 * ```
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Apply a transformation function to any errors returned by the decoder.
 */ parcelHelpers.export(exports, "map_errors", ()=>map_errors);
/**
 * Create a new decoder based upon the value of a previous decoder.
 *
 * This may be useful to run one previous decoder to use in further decoding.
 */ parcelHelpers.export(exports, "then$", ()=>then$);
/**
 * Create a new decoder from several other decoders. Each of the inner
 * decoders is run in turn, and the value from the first to succeed is used.
 *
 * If no decoder succeeds then the errors from the first decoder are used.
 * If you wish for different errors then you may wish to use the
 * `collapse_errors` or `map_errors` functions.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = decode.one_of(decode.string, or: [
 *   decode.int |> decode.map(int.to_string),
 *   decode.float |> decode.map(float.to_string),
 * ])
 * assert decode.run(dynamic.int(1000), decoder) == Ok("1000")
 * ```
 */ parcelHelpers.export(exports, "one_of", ()=>one_of);
/**
 * Create a decoder that can refer to itself, useful for decoding deeply
 * nested data.
 *
 * Attempting to create a recursive decoder without this function could result
 * in an infinite loop. If you are using `field` or other `use`able functions
 * then you may not need to use this function.
 *
 * ## Examples
 *
 * ```gleam
 * type Nested {
 *   Nested(List(Nested))
 *   Value(String)
 * }
 *
 * fn nested_decoder() -> decode.Decoder(Nested) {
 *   use <- decode.recursive
 *   decode.one_of(decode.string |> decode.map(Value), [
 *     decode.list(nested_decoder()) |> decode.map(Nested),
 *   ])
 * }
 * ```
 */ parcelHelpers.export(exports, "recursive", ()=>recursive);
/**
 * A decoder that decodes nullable values of a type decoded by with a given
 * decoder.
 *
 * This function can handle common representations of null on all runtimes, such as
 * `nil`, `null`, and `undefined` on Erlang, and `undefined` and `null` on
 * JavaScript.
 *
 * ## Examples
 *
 * ```gleam
 * let result = decode.run(dynamic.int(100), decode.optional(decode.int))
 * assert result == Ok(option.Some(100))
 * ```
 *
 * ```gleam
 * let result = decode.run(dynamic.nil(), decode.optional(decode.int))
 * assert result == Ok(option.None)
 * ```
 */ parcelHelpers.export(exports, "optional", ()=>optional);
/**
 * Construct a decode error for some unexpected dynamic data.
 */ parcelHelpers.export(exports, "decode_error", ()=>decode_error);
/**
 * Replace all errors produced by a decoder with one single error for a named
 * expected type.
 *
 * This function may be useful if you wish to simplify errors before
 * presenting them to a user, particularly when using the `one_of` function.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = decode.string |> decode.collapse_errors("MyThing")
 * let result = decode.run(dynamic.int(1000), decoder)
 * assert result == Error([DecodeError("MyThing", "Int", [])])
 * ```
 */ parcelHelpers.export(exports, "collapse_errors", ()=>collapse_errors);
/**
 * Define a decoder that always fails.
 *
 * The first parameter is a "placeholder" value, which is some default value that the
 * decoder uses internally in place of the value that would have been produced
 * if the decoder was successful. It doesn't matter what this value is, it is
 * never returned by the decoder or shown to the user, so pick some arbitrary
 * value. If it is an int you might pick `0`, if it is a list you might pick
 * `[]`.
 *
 * The second parameter is the name of the type that has failed to decode.
 *
 * ```gleam
 * decode.failure(User(name: "", score: 0, tags: []), expected: "User")
 * ```
 */ parcelHelpers.export(exports, "failure", ()=>failure);
/**
 * Create a decoder for a new data type from a decoding function.
 *
 * This function is used for new primitive types. For example, you might
 * define a decoder for Erlang's pid type.
 *
 * A default "placeholder" value is also required to make a decoder. When this
 * decoder is used as part of a larger decoder this placeholder value is used
 * so that the rest of the decoder can continue to run and
 * collect all decoding errors. It doesn't matter what this value is, it is
 * never returned by the decoder or shown to the user, so pick some arbitrary
 * value. If it is an int you might pick `0`, if it is a list you might pick
 * `[]`.
 *
 * If you were to make a decoder for the `Int` type (rather than using the
 * build-in `Int` decoder) you would define it like so:
 *
 * ```gleam
 * pub fn int_decoder() -> decode.Decoder(Int) {
 *   let default = ""
 *   decode.new_primitive_decoder("Int", int_from_dynamic)
 * }
 *
 * @external(erlang, "my_module", "int_from_dynamic")
 * fn int_from_dynamic(data: Int) -> Result(Int, Int)
 * ```
 *
 * ```erlang
 * -module(my_module).
 * -export([int_from_dynamic/1]).
 *
 * int_from_dynamic(Data) ->
 *     case is_integer(Data) of
 *         true -> {ok, Data};
 *         false -> {error, 0}
 *     end.
 * ```
 */ parcelHelpers.export(exports, "new_primitive_decoder", ()=>new_primitive_decoder);
/**
 * A decoder that decodes dicts where all keys and values are decoded with
 * given decoders.
 *
 * ## Examples
 *
 * ```gleam
 * let values = dynamic.properties([
 *   #(dynamic.string("one"), dynamic.int(1)),
 *   #(dynamic.string("two"), dynamic.int(2)),
 * ])
 *
 * let result =
 *   decode.run(values, decode.dict(decode.string, decode.int))
 * assert result == Ok(values)
 * ```
 */ parcelHelpers.export(exports, "dict", ()=>dict);
/**
 * A decoder that decodes lists where all elements are decoded with a given
 * decoder.
 *
 * ## Examples
 *
 * ```gleam
 * let result =
 *   [1, 2, 3]
 *   |> list.map(dynamic.int)
 *   |> dynamic.list
 *   |> decode.run(decode.list(of: decode.int))
 * assert result == Ok([1, 2, 3])
 * ```
 */ parcelHelpers.export(exports, "list", ()=>list);
/**
 * The same as [`field`](#field), except taking a path to the value rather
 * than a field name.
 *
 * This function will index into dictionaries with any key type, and if the key is
 * an int then it'll also index into Erlang tuples and JavaScript arrays, and
 * the first eight elements of Gleam lists.
 *
 * ## Examples
 *
 * ```gleam
 * let data = dynamic.properties([
 *   #(dynamic.string("data"), dynamic.properties([
 *     #(dynamic.string("email"), dynamic.string("lucy@example.com")),
 *     #(dynamic.string("name"), dynamic.string("Lucy")),
 *   ])
 * ])
 *
 * let decoder = {
 *   use name <- decode.subfield(["data", "name"], decode.string)
 *   use email <- decode.subfield(["data", "email"], decode.string)
 *   decode.success(SignUp(name: name, email: email))
 * }
 * let result = decode.run(data, decoder)
 * assert result == Ok(SignUp(name: "Lucy", email: "lucy@example.com"))
 * ```
 */ parcelHelpers.export(exports, "subfield", ()=>subfield);
/**
 * A decoder that decodes a value that is nested within other values. For
 * example, decoding a value that is within some deeply nested JSON objects.
 *
 * This function will index into dictionaries with any key type, and if the key is
 * an int then it'll also index into Erlang tuples and JavaScript arrays, and
 * the first eight elements of Gleam lists.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = decode.at(["one", "two"], decode.int)
 *
 * let data = dynamic.properties([
 *   #(dynamic.string("one"), dynamic.properties([
 *     #(dynamic.string("two"), dynamic.int(1000)),
 *   ]),
 * ])
 *
 * assert decode.run(data, decoder) == Ok(1000)
 * ```
 *
 * ```gleam
 * assert dynamic.nil()
 *   |> decode.run(decode.optional(decode.int))
 *   == Ok(option.None)
 * ```
 */ parcelHelpers.export(exports, "at", ()=>at);
/**
 * Run a decoder on a field of a `Dynamic` value, decoding the value if it is
 * of the desired type, or returning errors. An error is returned if there is
 * no field for the specified key.
 *
 * This function will index into dictionaries with any key type, and if the key is
 * an int then it'll also index into Erlang tuples and JavaScript arrays, and
 * the first eight elements of Gleam lists.
 *
 * ## Examples
 *
 * ```gleam
 * let data = dynamic.properties([
 *   #(dynamic.string("email"), dynamic.string("lucy@example.com")),
 *   #(dynamic.string("name"), dynamic.string("Lucy")),
 * ])
 *
 * let decoder = {
 *   use name <- decode.field("name", string)
 *   use email <- decode.field("email", string)
 *   decode.success(SignUp(name: name, email: email))
 * }
 *
 * let result = decode.run(data, decoder)
 * assert result == Ok(SignUp(name: "Lucy", email: "lucy@example.com"))
 * ```
 *
 * If you wish to decode a value that is more deeply nested within the dynamic
 * data, see [`subfield`](#subfield) and [`at`](#at).
 *
 * If you wish to return a default in the event that a field is not present,
 * see [`optional_field`](#optional_field) and / [`optionally_at`](#optionally_at).
 */ parcelHelpers.export(exports, "field", ()=>field);
/**
 * Run a decoder on a field of a `Dynamic` value, decoding the value if it is
 * of the desired type, or returning errors. The given default value is
 * returned if there is no field for the specified key.
 *
 * This function will index into dictionaries with any key type, and if the key is
 * an int then it'll also index into Erlang tuples and JavaScript arrays, and
 * the first eight elements of Gleam lists.
 *
 * ## Examples
 *
 * ```gleam
 * let data = dynamic.properties([
 *   #(dynamic.string("name"), dynamic.string("Lucy")),
 * ])
 *
 * let decoder = {
 *   use name <- decode.field("name", string)
 *   use email <- decode.optional_field("email", "n/a", string)
 *   decode.success(SignUp(name: name, email: email))
 * }
 *
 * let result = decode.run(data, decoder)
 * assert result == Ok(SignUp(name: "Lucy", email: "n/a"))
 * ```
 */ parcelHelpers.export(exports, "optional_field", ()=>optional_field);
/**
 * A decoder that decodes a value that is nested within other values. For
 * example, decoding a value that is within some deeply nested JSON objects.
 *
 * This function will index into dictionaries with any key type, and if the key is
 * an int then it'll also index into Erlang tuples and JavaScript arrays, and
 * the first eight elements of Gleam lists.
 *
 * ## Examples
 *
 * ```gleam
 * let decoder = decode.optionally_at(["one", "two"], 100, decode.int)
 *
 * let data = dynamic.properties([
 *   #(dynamic.string("one"), dynamic.properties([])),
 * ])
 *
 * assert decode.run(data, decoder) == Ok(100)
 * ```
 */ parcelHelpers.export(exports, "optionally_at", ()=>optionally_at);
var _gleamMjs = require("../../gleam.mjs");
var _bitArrayMjs = require("../../gleam/bit_array.mjs");
var _dictMjs = require("../../gleam/dict.mjs");
var _dynamicMjs = require("../../gleam/dynamic.mjs");
var _floatMjs = require("../../gleam/float.mjs");
var _intMjs = require("../../gleam/int.mjs");
var _listMjs = require("../../gleam/list.mjs");
var _optionMjs = require("../../gleam/option.mjs");
var _gleamStdlibMjs = require("../../gleam_stdlib.mjs");
class DecodeError extends (0, _gleamMjs.CustomType) {
    constructor(expected, found, path){
        super();
        this.expected = expected;
        this.found = found;
        this.path = path;
    }
}
const DecodeError$DecodeError = (expected, found, path)=>new DecodeError(expected, found, path);
const DecodeError$isDecodeError = (value)=>value instanceof DecodeError;
const DecodeError$DecodeError$expected = (value)=>value.expected;
const DecodeError$DecodeError$0 = (value)=>value.expected;
const DecodeError$DecodeError$found = (value)=>value.found;
const DecodeError$DecodeError$1 = (value)=>value.found;
const DecodeError$DecodeError$path = (value)=>value.path;
const DecodeError$DecodeError$2 = (value)=>value.path;
class Decoder extends (0, _gleamMjs.CustomType) {
    constructor(function$){
        super();
        this.function = function$;
    }
}
const dynamic = /* @__PURE__ */ new Decoder(decode_dynamic);
const bool = /* @__PURE__ */ new Decoder(decode_bool);
const int = /* @__PURE__ */ new Decoder(decode_int);
const float = /* @__PURE__ */ new Decoder(decode_float);
const bit_array = /* @__PURE__ */ new Decoder(decode_bit_array);
const string = /* @__PURE__ */ new Decoder(decode_string);
function run(data, decoder) {
    let $ = decoder.function(data);
    let maybe_invalid_data;
    let errors;
    maybe_invalid_data = $[0];
    errors = $[1];
    if (errors instanceof (0, _gleamMjs.Empty)) return new (0, _gleamMjs.Ok)(maybe_invalid_data);
    else return new (0, _gleamMjs.Error)(errors);
}
function success(data) {
    return new Decoder((_)=>{
        return [
            data,
            (0, _gleamMjs.toList)([])
        ];
    });
}
function decode_dynamic(data) {
    return [
        data,
        (0, _gleamMjs.toList)([])
    ];
}
function map(decoder, transformer) {
    return new Decoder((d)=>{
        let $ = decoder.function(d);
        let data;
        let errors;
        data = $[0];
        errors = $[1];
        return [
            transformer(data),
            errors
        ];
    });
}
function map_errors(decoder, transformer) {
    return new Decoder((d)=>{
        let $ = decoder.function(d);
        let data;
        let errors;
        data = $[0];
        errors = $[1];
        return [
            data,
            transformer(errors)
        ];
    });
}
function then$(decoder, next) {
    return new Decoder((dynamic_data)=>{
        let $ = decoder.function(dynamic_data);
        let data;
        let errors;
        data = $[0];
        errors = $[1];
        let decoder$1 = next(data);
        let $1 = decoder$1.function(dynamic_data);
        let layer;
        let data$1;
        layer = $1;
        data$1 = $1[0];
        if (errors instanceof (0, _gleamMjs.Empty)) return layer;
        else return [
            data$1,
            errors
        ];
    });
}
function run_decoders(loop$data, loop$failure, loop$decoders) {
    while(true){
        let data = loop$data;
        let failure = loop$failure;
        let decoders = loop$decoders;
        if (decoders instanceof (0, _gleamMjs.Empty)) return failure;
        else {
            let decoder = decoders.head;
            let decoders$1 = decoders.tail;
            let $ = decoder.function(data);
            let layer;
            let errors;
            layer = $;
            errors = $[1];
            if (errors instanceof (0, _gleamMjs.Empty)) return layer;
            else {
                loop$data = data;
                loop$failure = failure;
                loop$decoders = decoders$1;
            }
        }
    }
}
function one_of(first, alternatives) {
    return new Decoder((dynamic_data)=>{
        let $ = first.function(dynamic_data);
        let layer;
        let errors;
        layer = $;
        errors = $[1];
        if (errors instanceof (0, _gleamMjs.Empty)) return layer;
        else return run_decoders(dynamic_data, layer, alternatives);
    });
}
function recursive(inner) {
    return new Decoder((data)=>{
        let decoder = inner();
        return decoder.function(data);
    });
}
function optional(inner) {
    return new Decoder((data)=>{
        let $ = (0, _gleamStdlibMjs.is_null)(data);
        if ($) return [
            new _optionMjs.None(),
            (0, _gleamMjs.toList)([])
        ];
        else {
            let $1 = inner.function(data);
            let data$1;
            let errors;
            data$1 = $1[0];
            errors = $1[1];
            return [
                new _optionMjs.Some(data$1),
                errors
            ];
        }
    });
}
function decode_error(expected, found) {
    return (0, _gleamMjs.toList)([
        new DecodeError(expected, _dynamicMjs.classify(found), (0, _gleamMjs.toList)([]))
    ]);
}
function run_dynamic_function(data, name, f) {
    let $ = f(data);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let data$1 = $[0];
        return [
            data$1,
            (0, _gleamMjs.toList)([])
        ];
    } else {
        let placeholder = $[0];
        return [
            placeholder,
            (0, _gleamMjs.toList)([
                new DecodeError(name, _dynamicMjs.classify(data), (0, _gleamMjs.toList)([]))
            ])
        ];
    }
}
function decode_bool(data) {
    let $ = (0, _gleamMjs.isEqual)((0, _gleamStdlibMjs.identity)(true), data);
    if ($) return [
        true,
        (0, _gleamMjs.toList)([])
    ];
    else {
        let $1 = (0, _gleamMjs.isEqual)((0, _gleamStdlibMjs.identity)(false), data);
        if ($1) return [
            false,
            (0, _gleamMjs.toList)([])
        ];
        else return [
            false,
            decode_error("Bool", data)
        ];
    }
}
function decode_int(data) {
    return run_dynamic_function(data, "Int", (0, _gleamStdlibMjs.int));
}
function decode_float(data) {
    return run_dynamic_function(data, "Float", (0, _gleamStdlibMjs.float));
}
function decode_bit_array(data) {
    return run_dynamic_function(data, "BitArray", (0, _gleamStdlibMjs.bit_array));
}
function collapse_errors(decoder, name) {
    return new Decoder((dynamic_data)=>{
        let $ = decoder.function(dynamic_data);
        let layer;
        let data;
        let errors;
        layer = $;
        data = $[0];
        errors = $[1];
        if (errors instanceof (0, _gleamMjs.Empty)) return layer;
        else return [
            data,
            decode_error(name, dynamic_data)
        ];
    });
}
function failure(placeholder, name) {
    return new Decoder((d)=>{
        return [
            placeholder,
            decode_error(name, d)
        ];
    });
}
function new_primitive_decoder(name, decoding_function) {
    return new Decoder((d)=>{
        let $ = decoding_function(d);
        if ($ instanceof (0, _gleamMjs.Ok)) {
            let t = $[0];
            return [
                t,
                (0, _gleamMjs.toList)([])
            ];
        } else {
            let placeholder = $[0];
            return [
                placeholder,
                (0, _gleamMjs.toList)([
                    new DecodeError(name, _dynamicMjs.classify(d), (0, _gleamMjs.toList)([]))
                ])
            ];
        }
    });
}
function decode_string(data) {
    return run_dynamic_function(data, "String", (0, _gleamStdlibMjs.string));
}
function path_segment_to_string(key) {
    let decoder = one_of(string, (0, _gleamMjs.toList)([
        (()=>{
            let _pipe = int;
            return map(_pipe, _intMjs.to_string);
        })(),
        (()=>{
            let _pipe = float;
            return map(_pipe, _floatMjs.to_string);
        })()
    ]));
    let $ = run(key, decoder);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let key$1 = $[0];
        return key$1;
    } else return "<" + _dynamicMjs.classify(key) + ">";
}
function fold_dict(acc, key, value, key_decoder, value_decoder) {
    let $ = key_decoder(key);
    let $1 = $[1];
    if ($1 instanceof (0, _gleamMjs.Empty)) {
        let key_decoded = $[0];
        let $2 = value_decoder(value);
        let $3 = $2[1];
        if ($3 instanceof (0, _gleamMjs.Empty)) {
            let value$1 = $2[0];
            let dict$1 = _dictMjs.insert(acc[0], key_decoded, value$1);
            return [
                dict$1,
                acc[1]
            ];
        } else {
            let errors = $3;
            let key_identifier = path_segment_to_string(key);
            return push_path([
                _dictMjs.new$(),
                errors
            ], (0, _gleamMjs.toList)([
                key_identifier
            ]));
        }
    } else {
        let errors = $1;
        return push_path([
            _dictMjs.new$(),
            errors
        ], (0, _gleamMjs.toList)([
            "keys"
        ]));
    }
}
function dict(key, value) {
    return new Decoder((data)=>{
        let $ = (0, _gleamStdlibMjs.dict)(data);
        if ($ instanceof (0, _gleamMjs.Ok)) {
            let dict$1 = $[0];
            return _dictMjs.fold(dict$1, [
                _dictMjs.new$(),
                (0, _gleamMjs.toList)([])
            ], (a, k, v)=>{
                let $1 = a[1];
                if ($1 instanceof (0, _gleamMjs.Empty)) return fold_dict(a, k, v, key.function, value.function);
                else return a;
            });
        } else return [
            _dictMjs.new$(),
            decode_error("Dict", data)
        ];
    });
}
function list(inner) {
    return new Decoder((data)=>{
        return (0, _gleamStdlibMjs.list)(data, inner.function, (p, k)=>{
            return push_path(p, (0, _gleamMjs.toList)([
                k
            ]));
        }, 0, (0, _gleamMjs.toList)([]));
    });
}
function push_path(layer, path) {
    let path$1 = _listMjs.map(path, (key)=>{
        let _pipe = key;
        let _pipe$1 = (0, _gleamStdlibMjs.identity)(_pipe);
        return path_segment_to_string(_pipe$1);
    });
    let errors = _listMjs.map(layer[1], (error)=>{
        return new DecodeError(error.expected, error.found, _listMjs.append(path$1, error.path));
    });
    return [
        layer[0],
        errors
    ];
}
function index(loop$path, loop$position, loop$inner, loop$data, loop$handle_miss) {
    while(true){
        let path = loop$path;
        let position = loop$position;
        let inner = loop$inner;
        let data = loop$data;
        let handle_miss = loop$handle_miss;
        if (path instanceof (0, _gleamMjs.Empty)) {
            let _pipe = data;
            let _pipe$1 = inner(_pipe);
            return push_path(_pipe$1, _listMjs.reverse(position));
        } else {
            let key = path.head;
            let path$1 = path.tail;
            let $ = (0, _gleamStdlibMjs.index)(data, key);
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let $1 = $[0];
                if ($1 instanceof (0, _optionMjs.Some)) {
                    let data$1 = $1[0];
                    loop$path = path$1;
                    loop$position = (0, _gleamMjs.prepend)(key, position);
                    loop$inner = inner;
                    loop$data = data$1;
                    loop$handle_miss = handle_miss;
                } else return handle_miss(data, (0, _gleamMjs.prepend)(key, position));
            } else {
                let kind = $[0];
                let $1 = inner(data);
                let default$;
                default$ = $1[0];
                let _pipe = [
                    default$,
                    (0, _gleamMjs.toList)([
                        new DecodeError(kind, _dynamicMjs.classify(data), (0, _gleamMjs.toList)([]))
                    ])
                ];
                return push_path(_pipe, _listMjs.reverse(position));
            }
        }
    }
}
function subfield(field_path, field_decoder, next) {
    return new Decoder((data)=>{
        let $ = index(field_path, (0, _gleamMjs.toList)([]), field_decoder.function, data, (data, position)=>{
            let $1 = field_decoder.function(data);
            let default$;
            default$ = $1[0];
            let _pipe = [
                default$,
                (0, _gleamMjs.toList)([
                    new DecodeError("Field", "Nothing", (0, _gleamMjs.toList)([]))
                ])
            ];
            return push_path(_pipe, _listMjs.reverse(position));
        });
        let out;
        let errors1;
        out = $[0];
        errors1 = $[1];
        let $1 = next(out).function(data);
        let out$1;
        let errors2;
        out$1 = $1[0];
        errors2 = $1[1];
        return [
            out$1,
            _listMjs.append(errors1, errors2)
        ];
    });
}
function at(path, inner) {
    return new Decoder((data)=>{
        return index(path, (0, _gleamMjs.toList)([]), inner.function, data, (data, position)=>{
            let $ = inner.function(data);
            let default$;
            default$ = $[0];
            let _pipe = [
                default$,
                (0, _gleamMjs.toList)([
                    new DecodeError("Field", "Nothing", (0, _gleamMjs.toList)([]))
                ])
            ];
            return push_path(_pipe, _listMjs.reverse(position));
        });
    });
}
function field(field_name, field_decoder, next) {
    return subfield((0, _gleamMjs.toList)([
        field_name
    ]), field_decoder, next);
}
function optional_field(key, default$, field_decoder, next) {
    return new Decoder((data)=>{
        let _block;
        let _block$1;
        let $1 = (0, _gleamStdlibMjs.index)(data, key);
        if ($1 instanceof (0, _gleamMjs.Ok)) {
            let $2 = $1[0];
            if ($2 instanceof (0, _optionMjs.Some)) {
                let data$1 = $2[0];
                _block$1 = field_decoder.function(data$1);
            } else _block$1 = [
                default$,
                (0, _gleamMjs.toList)([])
            ];
        } else {
            let kind = $1[0];
            _block$1 = [
                default$,
                (0, _gleamMjs.toList)([
                    new DecodeError(kind, _dynamicMjs.classify(data), (0, _gleamMjs.toList)([]))
                ])
            ];
        }
        let _pipe = _block$1;
        _block = push_path(_pipe, (0, _gleamMjs.toList)([
            key
        ]));
        let $ = _block;
        let out;
        let errors1;
        out = $[0];
        errors1 = $[1];
        let $2 = next(out).function(data);
        let out$1;
        let errors2;
        out$1 = $2[0];
        errors2 = $2[1];
        return [
            out$1,
            _listMjs.append(errors1, errors2)
        ];
    });
}
function optionally_at(path, default$, inner) {
    return new Decoder((data)=>{
        return index(path, (0, _gleamMjs.toList)([]), inner.function, data, (_, _1)=>{
            return [
                default$,
                (0, _gleamMjs.toList)([])
            ];
        });
    });
}

},{"../../gleam.mjs":"aiPrb","../../gleam/bit_array.mjs":"69HLR","../../gleam/dict.mjs":"b8yrU","../../gleam/dynamic.mjs":"iAWCk","../../gleam/float.mjs":"9bPI9","../../gleam/int.mjs":"32hLf","../../gleam/list.mjs":"8dUwY","../../gleam/option.mjs":"aWtoH","../../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"69HLR":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "base16_decode", ()=>(0, _gleamStdlibMjs.base16_decode));
parcelHelpers.export(exports, "base16_encode", ()=>(0, _gleamStdlibMjs.base16_encode));
parcelHelpers.export(exports, "base64_encode", ()=>(0, _gleamStdlibMjs.base64_encode));
parcelHelpers.export(exports, "bit_size", ()=>(0, _gleamStdlibMjs.bit_array_bit_size));
parcelHelpers.export(exports, "byte_size", ()=>(0, _gleamStdlibMjs.bit_array_byte_size));
parcelHelpers.export(exports, "concat", ()=>(0, _gleamStdlibMjs.bit_array_concat));
parcelHelpers.export(exports, "from_string", ()=>(0, _gleamStdlibMjs.bit_array_from_string));
parcelHelpers.export(exports, "pad_to_bytes", ()=>(0, _gleamStdlibMjs.bit_array_pad_to_bytes));
parcelHelpers.export(exports, "slice", ()=>(0, _gleamStdlibMjs.bit_array_slice));
parcelHelpers.export(exports, "starts_with", ()=>(0, _gleamStdlibMjs.bit_array_starts_with));
parcelHelpers.export(exports, "to_string", ()=>(0, _gleamStdlibMjs.bit_array_to_string));
/**
 * Creates a new bit array by joining two bit arrays.
 *
 * ## Examples
 *
 * ```gleam
 * assert append(to: from_string("butter"), suffix: from_string("fly"))
 *   == from_string("butterfly")
 * ```
 */ parcelHelpers.export(exports, "append", ()=>append);
/**
 * Decodes a base 64 encoded string into a `BitArray`.
 */ parcelHelpers.export(exports, "base64_decode", ()=>base64_decode);
/**
 * Encodes a `BitArray` into a base 64 encoded string with URL and filename
 * safe alphabet.
 *
 * If the bit array does not contain a whole number of bytes then it is padded
 * with zero bits prior to being encoded.
 */ parcelHelpers.export(exports, "base64_url_encode", ()=>base64_url_encode);
/**
 * Decodes a base 64 encoded string with URL and filename safe alphabet into a
 * `BitArray`.
 */ parcelHelpers.export(exports, "base64_url_decode", ()=>base64_url_decode);
/**
 * Converts a bit array to a string containing the decimal value of each byte.
 *
 * Use this over `string.inspect` when you have a bit array you want printed
 * in the array syntax even if it is valid UTF-8.
 *
 * ## Examples
 *
 * ```gleam
 * assert inspect(<<0, 20, 0x20, 255>>) == "<<0, 20, 32, 255>>"
 * ```
 *
 * ```gleam
 * assert inspect(<<100, 5:3>>) == "<<100, 5:size(3)>>"
 * ```
 */ parcelHelpers.export(exports, "inspect", ()=>inspect);
/**
 * Compare two bit arrays as sequences of bytes.
 *
 * ## Examples
 *
 * ```gleam
 * assert compare(<<1>>, <<2>>) == Lt
 * ```
 *
 * ```gleam
 * assert compare(<<"AB":utf8>>, <<"AA":utf8>>) == Gt
 * ```
 *
 * ```gleam
 * assert compare(<<1, 2:size(2)>>, with: <<1, 2:size(2)>>) == Eq
 * ```
 */ parcelHelpers.export(exports, "compare", ()=>compare);
/**
 * Tests to see whether a bit array is valid UTF-8.
 */ parcelHelpers.export(exports, "is_utf8", ()=>is_utf8);
var _gleamMjs = require("../gleam.mjs");
var _intMjs = require("../gleam/int.mjs");
var _orderMjs = require("../gleam/order.mjs");
var _stringMjs = require("../gleam/string.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
function append(first, second) {
    return (0, _gleamStdlibMjs.bit_array_concat)((0, _gleamMjs.toList)([
        first,
        second
    ]));
}
function base64_decode(encoded) {
    let _block;
    let $ = (0, _gleamStdlibMjs.bit_array_byte_size)((0, _gleamStdlibMjs.bit_array_from_string)(encoded)) % 4;
    if ($ === 0) _block = encoded;
    else {
        let n = $;
        _block = _stringMjs.append(encoded, _stringMjs.repeat("=", 4 - n));
    }
    let padded = _block;
    return (0, _gleamStdlibMjs.base64_decode)(padded);
}
function base64_url_encode(input, padding) {
    let _pipe = input;
    let _pipe$1 = (0, _gleamStdlibMjs.base64_encode)(_pipe, padding);
    let _pipe$2 = _stringMjs.replace(_pipe$1, "+", "-");
    return _stringMjs.replace(_pipe$2, "/", "_");
}
function base64_url_decode(encoded) {
    let _pipe = encoded;
    let _pipe$1 = _stringMjs.replace(_pipe, "-", "+");
    let _pipe$2 = _stringMjs.replace(_pipe$1, "_", "/");
    return base64_decode(_pipe$2);
}
function inspect_loop(loop$input, loop$accumulator) {
    while(true){
        let input = loop$input;
        let accumulator = loop$accumulator;
        if (input.bitSize === 0) return accumulator;
        else if (input.bitSize === 1) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 1, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(1)";
        } else if (input.bitSize === 2) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 2, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(2)";
        } else if (input.bitSize === 3) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 3, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(3)";
        } else if (input.bitSize === 4) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 4, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(4)";
        } else if (input.bitSize === 5) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 5, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(5)";
        } else if (input.bitSize === 6) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 6, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(6)";
        } else if (input.bitSize === 7) {
            let x = (0, _gleamMjs.bitArraySliceToInt)(input, 0, 7, true, false);
            return accumulator + _intMjs.to_string(x) + ":size(7)";
        } else if (input.bitSize >= 8) {
            let x = input.byteAt(0);
            let rest = (0, _gleamMjs.bitArraySlice)(input, 8);
            let _block;
            if (rest.bitSize === 0) _block = "";
            else _block = ", ";
            let suffix = _block;
            let accumulator$1 = accumulator + _intMjs.to_string(x) + suffix;
            loop$input = rest;
            loop$accumulator = accumulator$1;
        } else return accumulator;
    }
}
function inspect(input) {
    return inspect_loop(input, "<<") + ">>";
}
function compare(loop$a, loop$b) {
    while(true){
        let a = loop$a;
        let b = loop$b;
        if (a.bitSize >= 8) {
            if (b.bitSize >= 8) {
                let first_byte = a.byteAt(0);
                let first_rest = (0, _gleamMjs.bitArraySlice)(a, 8);
                let second_byte = b.byteAt(0);
                let second_rest = (0, _gleamMjs.bitArraySlice)(b, 8);
                let f = first_byte;
                let s = second_byte;
                if (f > s) return new _orderMjs.Gt();
                else {
                    let f = first_byte;
                    let s = second_byte;
                    if (f < s) return new _orderMjs.Lt();
                    else {
                        loop$a = first_rest;
                        loop$b = second_rest;
                    }
                }
            } else if (b.bitSize === 0) return new _orderMjs.Gt();
            else {
                let first = a;
                let second = b;
                let $ = (0, _gleamStdlibMjs.bit_array_to_int_and_size)(first);
                let $1 = (0, _gleamStdlibMjs.bit_array_to_int_and_size)(second);
                let a$1 = $[0];
                let b$1 = $1[0];
                if (a$1 > b$1) return new _orderMjs.Gt();
                else {
                    let a$1 = $[0];
                    let b$1 = $1[0];
                    if (a$1 < b$1) return new _orderMjs.Lt();
                    else {
                        let size_a = $[1];
                        let size_b = $1[1];
                        if (size_a > size_b) return new _orderMjs.Gt();
                        else {
                            let size_a = $[1];
                            let size_b = $1[1];
                            if (size_a < size_b) return new _orderMjs.Lt();
                            else return new _orderMjs.Eq();
                        }
                    }
                }
            }
        } else if (b.bitSize === 0) {
            if (a.bitSize === 0) return new _orderMjs.Eq();
            else return new _orderMjs.Gt();
        } else if (a.bitSize === 0) return new _orderMjs.Lt();
        else {
            let first = a;
            let second = b;
            let $ = (0, _gleamStdlibMjs.bit_array_to_int_and_size)(first);
            let $1 = (0, _gleamStdlibMjs.bit_array_to_int_and_size)(second);
            let a$1 = $[0];
            let b$1 = $1[0];
            if (a$1 > b$1) return new _orderMjs.Gt();
            else {
                let a$1 = $[0];
                let b$1 = $1[0];
                if (a$1 < b$1) return new _orderMjs.Lt();
                else {
                    let size_a = $[1];
                    let size_b = $1[1];
                    if (size_a > size_b) return new _orderMjs.Gt();
                    else {
                        let size_a = $[1];
                        let size_b = $1[1];
                        if (size_a < size_b) return new _orderMjs.Lt();
                        else return new _orderMjs.Eq();
                    }
                }
            }
        }
    }
}
function is_utf8(bits) {
    return is_utf8_loop(bits);
}
function is_utf8_loop(bits) {
    let $ = (0, _gleamStdlibMjs.bit_array_to_string)(bits);
    if ($ instanceof (0, _gleamMjs.Ok)) return true;
    else return false;
}

},{"../gleam.mjs":"aiPrb","../gleam/int.mjs":"32hLf","../gleam/order.mjs":"eYj92","../gleam/string.mjs":"aB8qb","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"32hLf":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "bitwise_and", ()=>(0, _gleamStdlibMjs.bitwise_and));
parcelHelpers.export(exports, "bitwise_exclusive_or", ()=>(0, _gleamStdlibMjs.bitwise_exclusive_or));
parcelHelpers.export(exports, "bitwise_not", ()=>(0, _gleamStdlibMjs.bitwise_not));
parcelHelpers.export(exports, "bitwise_or", ()=>(0, _gleamStdlibMjs.bitwise_or));
parcelHelpers.export(exports, "bitwise_shift_left", ()=>(0, _gleamStdlibMjs.bitwise_shift_left));
parcelHelpers.export(exports, "bitwise_shift_right", ()=>(0, _gleamStdlibMjs.bitwise_shift_right));
parcelHelpers.export(exports, "parse", ()=>(0, _gleamStdlibMjs.parse_int));
parcelHelpers.export(exports, "to_float", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "to_string", ()=>(0, _gleamStdlibMjs.to_string));
/**
 * Returns the absolute value of the input.
 *
 * ## Examples
 *
 * ```gleam
 * assert absolute_value(-12) == 12
 * ```
 *
 * ```gleam
 * assert absolute_value(10) == 10
 * ```
 */ parcelHelpers.export(exports, "absolute_value", ()=>absolute_value);
/**
 * Parses a given string as an int in a given base if possible.
 * Supports only bases 2 to 36, for values outside of which this function returns an `Error(Nil)`.
 *
 * ## Examples
 *
 * ```gleam
 * assert base_parse("10", 2) == Ok(2)
 * ```
 *
 * ```gleam
 * assert base_parse("30", 16) == Ok(48)
 * ```
 *
 * ```gleam
 * assert base_parse("1C", 36) == Ok(48)
 * ```
 *
 * ```gleam
 * assert base_parse("48", 1) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert base_parse("48", 37) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "base_parse", ()=>base_parse);
/**
 * Prints a given int to a string using the base number provided.
 * Supports only bases 2 to 36, for values outside of which this function returns an `Error(Nil)`.
 * For common bases (2, 8, 16, 36), use the `to_baseN` functions.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_base_string(2, 2) == Ok("10")
 * ```
 *
 * ```gleam
 * assert to_base_string(48, 16) == Ok("30")
 * ```
 *
 * ```gleam
 * assert to_base_string(48, 36) == Ok("1C")
 * ```
 *
 * ```gleam
 * assert to_base_string(48, 1) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert to_base_string(48, 37) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "to_base_string", ()=>to_base_string);
/**
 * Prints a given int to a string using base-2.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_base2(2) == "10"
 * ```
 */ parcelHelpers.export(exports, "to_base2", ()=>to_base2);
/**
 * Prints a given int to a string using base-8.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_base8(15) == "17"
 * ```
 */ parcelHelpers.export(exports, "to_base8", ()=>to_base8);
/**
 * Prints a given int to a string using base-16.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_base16(48) == "30"
 * ```
 */ parcelHelpers.export(exports, "to_base16", ()=>to_base16);
/**
 * Prints a given int to a string using base-36.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_base36(48) == "1C"
 * ```
 */ parcelHelpers.export(exports, "to_base36", ()=>to_base36);
/**
 * Returns the result of the base being raised to the power of the
 * exponent, as a `Float`.
 *
 * ## Examples
 *
 * ```gleam
 * assert power(2, -1.0) == Ok(0.5)
 * ```
 *
 * ```gleam
 * assert power(2, 2.0) == Ok(4.0)
 * ```
 *
 * ```gleam
 * assert power(8, 1.5) == Ok(22.627416997969522)
 * ```
 *
 * ```gleam
 * assert 4 |> power(of: 2.0) == Ok(16.0)
 * ```
 *
 * ```gleam
 * assert power(-1, 0.5) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "power", ()=>power);
/**
 * Returns the square root of the input as a `Float`.
 *
 * ## Examples
 *
 * ```gleam
 * assert square_root(4) == Ok(2.0)
 * ```
 *
 * ```gleam
 * assert square_root(-16) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "square_root", ()=>square_root);
/**
 * Compares two ints, returning an order.
 *
 * ## Examples
 *
 * ```gleam
 * assert compare(2, 3) == Lt
 * ```
 *
 * ```gleam
 * assert compare(4, 3) == Gt
 * ```
 *
 * ```gleam
 * assert compare(3, 3) == Eq
 * ```
 */ parcelHelpers.export(exports, "compare", ()=>compare);
/**
 * Compares two ints, returning the smaller of the two.
 *
 * ## Examples
 *
 * ```gleam
 * assert min(2, 3) == 2
 * ```
 */ parcelHelpers.export(exports, "min", ()=>min);
/**
 * Compares two ints, returning the larger of the two.
 *
 * ## Examples
 *
 * ```gleam
 * assert max(2, 3) == 3
 * ```
 */ parcelHelpers.export(exports, "max", ()=>max);
/**
 * Restricts an int between two bounds.
 *
 * Note: If the `min` argument is larger than the `max` argument then they
 * will be swapped, so the minimum bound is always lower than the maximum
 * bound.
 *
 * ## Examples
 *
 * ```gleam
 * assert clamp(40, min: 50, max: 60) == 50
 * ```
 *
 * ```gleam
 * assert clamp(40, min: 50, max: 30) == 40
 * ```
 */ parcelHelpers.export(exports, "clamp", ()=>clamp);
/**
 * Returns whether the value provided is even.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_even(2)
 * ```
 *
 * ```gleam
 * assert !is_even(3)
 * ```
 */ parcelHelpers.export(exports, "is_even", ()=>is_even);
/**
 * Returns whether the value provided is odd.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_odd(3)
 * ```
 *
 * ```gleam
 * assert !is_odd(2)
 * ```
 */ parcelHelpers.export(exports, "is_odd", ()=>is_odd);
/**
 * Returns the negative of the value provided.
 *
 * ## Examples
 *
 * ```gleam
 * assert negate(1) == -1
 * ```
 */ parcelHelpers.export(exports, "negate", ()=>negate);
/**
 * Sums a list of ints.
 *
 * ## Example
 *
 * ```gleam
 * assert sum([1, 2, 3]) == 6
 * ```
 */ parcelHelpers.export(exports, "sum", ()=>sum);
/**
 * Multiplies a list of ints and returns the product.
 *
 * ## Example
 *
 * ```gleam
 * assert product([2, 3, 4]) == 24
 * ```
 */ parcelHelpers.export(exports, "product", ()=>product);
/**
 * Generates a random int between zero and the given maximum.
 *
 * The lower number is inclusive, the upper number is exclusive.
 *
 * ## Examples
 *
 * ```gleam
 * random(10)
 * // -> 4
 * ```
 *
 * ```gleam
 * random(1)
 * // -> 0
 * ```
 *
 * ```gleam
 * random(-1)
 * // -> -1
 * ```
 */ parcelHelpers.export(exports, "random", ()=>random);
/**
 * Performs a truncated integer division.
 *
 * Returns division of the inputs as a `Result`: If the given divisor equals
 * `0`, this function returns an `Error`.
 *
 * ## Examples
 *
 * ```gleam
 * assert divide(0, 1) == Ok(0)
 * ```
 *
 * ```gleam
 * assert divide(1, 0) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert divide(5, 2) == Ok(2)
 * ```
 *
 * ```gleam
 * assert divide(-99, 2) == Ok(-49)
 * ```
 */ parcelHelpers.export(exports, "divide", ()=>divide);
/**
 * Computes the remainder of an integer division of inputs as a `Result`.
 *
 * Returns division of the inputs as a `Result`: If the given divisor equals
 * `0`, this function returns an `Error`.
 *
 * Most of the time you will want to use the `%` operator instead of this
 * function.
 *
 * ## Examples
 *
 * ```gleam
 * assert remainder(3, 2) == Ok(1)
 * ```
 *
 * ```gleam
 * assert remainder(1, 0) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert remainder(10, -1) == Ok(0)
 * ```
 *
 * ```gleam
 * assert remainder(13, by: 3) == Ok(1)
 * ```
 *
 * ```gleam
 * assert remainder(-13, by: 3) == Ok(-1)
 * ```
 *
 * ```gleam
 * assert remainder(13, by: -3) == Ok(1)
 * ```
 *
 * ```gleam
 * assert remainder(-13, by: -3) == Ok(-1)
 * ```
 */ parcelHelpers.export(exports, "remainder", ()=>remainder);
/**
 * Computes the modulo of an integer division of inputs as a `Result`.
 *
 * Returns division of the inputs as a `Result`: If the given divisor equals
 * `0`, this function returns an `Error`.
 *
 * Note that this is different from `int.remainder` and `%` in that the
 * computed value will always have the same sign as the `divisor`.
 *
 * ## Examples
 *
 * ```gleam
 * assert modulo(3, 2) == Ok(1)
 * ```
 *
 * ```gleam
 * assert modulo(1, 0) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert modulo(10, -1) == Ok(0)
 * ```
 *
 * ```gleam
 * assert modulo(13, by: 3) == Ok(1)
 * ```
 *
 * ```gleam
 * assert modulo(-13, by: 3) == Ok(2)
 * ```
 *
 * ```gleam
 * assert modulo(13, by: -3) == Ok(-2)
 * ```
 */ parcelHelpers.export(exports, "modulo", ()=>modulo);
/**
 * Performs a *floored* integer division, which means that the result will
 * always be rounded towards negative infinity.
 *
 * If you want to perform truncated integer division (rounding towards zero),
 * use `int.divide()` or the `/` operator instead.
 *
 * Returns division of the inputs as a `Result`: If the given divisor equals
 * `0`, this function returns an `Error`.
 *
 * ## Examples
 *
 * ```gleam
 * assert floor_divide(1, 0) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert floor_divide(5, 2) == Ok(2)
 * ```
 *
 * ```gleam
 * assert floor_divide(6, -4) == Ok(-2)
 * ```
 *
 * ```gleam
 * assert floor_divide(-99, 2) == Ok(-50)
 * ```
 */ parcelHelpers.export(exports, "floor_divide", ()=>floor_divide);
/**
 * Adds two integers together.
 *
 * It's the function equivalent of the `+` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert add(1, 2) == 3
 * ```
 *
 * ```gleam
 * import gleam/list
 * assert list.fold([1, 2, 3], 0, add) == 6
 * ```
 *
 * ```gleam
 * assert 3 |> add(2) == 5
 * ```
 */ parcelHelpers.export(exports, "add", ()=>add);
/**
 * Multiplies two integers together.
 *
 * It's the function equivalent of the `*` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert multiply(2, 4) == 8
 * ```
 *
 * ```gleam
 * import gleam/list
 *
 * assert list.fold([2, 3, 4], 1, multiply) == 24
 * ```
 *
 * ```gleam
 * assert 3 |> multiply(2) == 6
 * ```
 */ parcelHelpers.export(exports, "multiply", ()=>multiply);
/**
 * Subtracts one int from another.
 *
 * It's the function equivalent of the `-` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert subtract(3, 1) == 2
 * ```
 *
 * ```gleam
 * import gleam/list
 *
 * assert list.fold([1, 2, 3], 10, subtract) == 4
 * ```
 *
 * ```gleam
 * assert 3 |> subtract(2) == 1
 * ```
 *
 * ```gleam
 * assert 3 |> subtract(2, _) == -1
 * ```
 */ parcelHelpers.export(exports, "subtract", ()=>subtract);
/**
 * Run a function for each int between ints `from` and `to`.
 *
 * `from` is inclusive, and `to` is exclusive.
 *
 * ## Examples
 *
 * ```gleam
 * assert
 *   range(from: 0, to: 3, with: "", run: fn(acc, i) {
 *     acc <> to_string(i)
 *   })
 *   == "012"
 * ```
 *
 * ```gleam
 * assert range(from: 1, to: -2, with: [], run: list.prepend) == [-1, 0, 1]
 * ```
 */ parcelHelpers.export(exports, "range", ()=>range);
var _gleamMjs = require("../gleam.mjs");
var _floatMjs = require("../gleam/float.mjs");
var _orderMjs = require("../gleam/order.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
function absolute_value(x) {
    let $ = x >= 0;
    if ($) return x;
    else return x * -1;
}
function base_parse(string, base) {
    let $ = base >= 2 && base <= 36;
    if ($) return (0, _gleamStdlibMjs.int_from_base_string)(string, base);
    else return new (0, _gleamMjs.Error)(undefined);
}
function to_base_string(x, base) {
    let $ = base >= 2 && base <= 36;
    if ($) return new (0, _gleamMjs.Ok)((0, _gleamStdlibMjs.int_to_base_string)(x, base));
    else return new (0, _gleamMjs.Error)(undefined);
}
function to_base2(x) {
    return (0, _gleamStdlibMjs.int_to_base_string)(x, 2);
}
function to_base8(x) {
    return (0, _gleamStdlibMjs.int_to_base_string)(x, 8);
}
function to_base16(x) {
    return (0, _gleamStdlibMjs.int_to_base_string)(x, 16);
}
function to_base36(x) {
    return (0, _gleamStdlibMjs.int_to_base_string)(x, 36);
}
function power(base, exponent) {
    let _pipe = base;
    let _pipe$1 = (0, _gleamStdlibMjs.identity)(_pipe);
    return _floatMjs.power(_pipe$1, exponent);
}
function square_root(x) {
    let _pipe = x;
    let _pipe$1 = (0, _gleamStdlibMjs.identity)(_pipe);
    return _floatMjs.square_root(_pipe$1);
}
function compare(a, b) {
    let $ = a === b;
    if ($) return new _orderMjs.Eq();
    else {
        let $1 = a < b;
        if ($1) return new _orderMjs.Lt();
        else return new _orderMjs.Gt();
    }
}
function min(a, b) {
    let $ = a < b;
    if ($) return a;
    else return b;
}
function max(a, b) {
    let $ = a > b;
    if ($) return a;
    else return b;
}
function clamp(x, min_bound, max_bound) {
    let $ = min_bound >= max_bound;
    if ($) {
        let _pipe = x;
        let _pipe$1 = min(_pipe, min_bound);
        return max(_pipe$1, max_bound);
    } else {
        let _pipe = x;
        let _pipe$1 = min(_pipe, max_bound);
        return max(_pipe$1, min_bound);
    }
}
function is_even(x) {
    return x % 2 === 0;
}
function is_odd(x) {
    return x % 2 !== 0;
}
function negate(x) {
    return -1 * x;
}
function sum_loop(loop$numbers, loop$initial) {
    while(true){
        let numbers = loop$numbers;
        let initial = loop$initial;
        if (numbers instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first = numbers.head;
            let rest = numbers.tail;
            loop$numbers = rest;
            loop$initial = first + initial;
        }
    }
}
function sum(numbers) {
    return sum_loop(numbers, 0);
}
function product_loop(loop$numbers, loop$initial) {
    while(true){
        let numbers = loop$numbers;
        let initial = loop$initial;
        if (numbers instanceof (0, _gleamMjs.Empty)) return initial;
        else {
            let first = numbers.head;
            let rest = numbers.tail;
            loop$numbers = rest;
            loop$initial = first * initial;
        }
    }
}
function product(numbers) {
    return product_loop(numbers, 1);
}
function random(max) {
    let _pipe = _floatMjs.random() * (0, _gleamStdlibMjs.identity)(max);
    let _pipe$1 = _floatMjs.floor(_pipe);
    return _floatMjs.round(_pipe$1);
}
function divide(dividend, divisor) {
    if (divisor === 0) return new (0, _gleamMjs.Error)(undefined);
    else {
        let divisor$1 = divisor;
        return new (0, _gleamMjs.Ok)((0, _gleamMjs.divideInt)(dividend, divisor$1));
    }
}
function remainder(dividend, divisor) {
    if (divisor === 0) return new (0, _gleamMjs.Error)(undefined);
    else {
        let divisor$1 = divisor;
        return new (0, _gleamMjs.Ok)((0, _gleamMjs.remainderInt)(dividend, divisor$1));
    }
}
function modulo(dividend, divisor) {
    if (divisor === 0) return new (0, _gleamMjs.Error)(undefined);
    else {
        let remainder$1 = (0, _gleamMjs.remainderInt)(dividend, divisor);
        let $ = remainder$1 * divisor < 0;
        if ($) return new (0, _gleamMjs.Ok)(remainder$1 + divisor);
        else return new (0, _gleamMjs.Ok)(remainder$1);
    }
}
function floor_divide(dividend, divisor) {
    if (divisor === 0) return new (0, _gleamMjs.Error)(undefined);
    else {
        let divisor$1 = divisor;
        let $ = dividend * divisor$1 < 0 && (0, _gleamMjs.remainderInt)(dividend, divisor$1) !== 0;
        if ($) return new (0, _gleamMjs.Ok)((0, _gleamMjs.divideInt)(dividend, divisor$1) - 1);
        else return new (0, _gleamMjs.Ok)((0, _gleamMjs.divideInt)(dividend, divisor$1));
    }
}
function add(a, b) {
    return a + b;
}
function multiply(a, b) {
    return a * b;
}
function subtract(a, b) {
    return a - b;
}
function range_loop(loop$current, loop$stop, loop$increment, loop$acc, loop$reducer) {
    while(true){
        let current = loop$current;
        let stop = loop$stop;
        let increment = loop$increment;
        let acc = loop$acc;
        let reducer = loop$reducer;
        let $ = current === stop;
        if ($) return acc;
        else {
            let acc$1 = reducer(acc, current);
            let current$1 = current + increment;
            loop$current = current$1;
            loop$stop = stop;
            loop$increment = increment;
            loop$acc = acc$1;
            loop$reducer = reducer;
        }
    }
}
function range(start, stop, acc, reducer) {
    let _block;
    let $ = start < stop;
    if ($) _block = 1;
    else _block = -1;
    let increment = _block;
    return range_loop(start, stop, increment, acc, reducer);
}

},{"../gleam.mjs":"aiPrb","../gleam/float.mjs":"9bPI9","../gleam/order.mjs":"eYj92","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"aB8qb":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "byte_size", ()=>(0, _gleamStdlibMjs.byte_size));
parcelHelpers.export(exports, "contains", ()=>(0, _gleamStdlibMjs.contains_string));
parcelHelpers.export(exports, "crop", ()=>(0, _gleamStdlibMjs.crop_string));
parcelHelpers.export(exports, "ends_with", ()=>(0, _gleamStdlibMjs.ends_with));
parcelHelpers.export(exports, "from_utf_codepoints", ()=>(0, _gleamStdlibMjs.utf_codepoint_list_to_string));
parcelHelpers.export(exports, "length", ()=>(0, _gleamStdlibMjs.string_length));
parcelHelpers.export(exports, "lowercase", ()=>(0, _gleamStdlibMjs.lowercase));
parcelHelpers.export(exports, "pop_grapheme", ()=>(0, _gleamStdlibMjs.pop_grapheme));
parcelHelpers.export(exports, "split_once", ()=>(0, _gleamStdlibMjs.split_once));
parcelHelpers.export(exports, "starts_with", ()=>(0, _gleamStdlibMjs.starts_with));
parcelHelpers.export(exports, "to_graphemes", ()=>(0, _gleamStdlibMjs.graphemes));
parcelHelpers.export(exports, "trim_end", ()=>(0, _gleamStdlibMjs.trim_end));
parcelHelpers.export(exports, "trim_start", ()=>(0, _gleamStdlibMjs.trim_start));
parcelHelpers.export(exports, "uppercase", ()=>(0, _gleamStdlibMjs.uppercase));
parcelHelpers.export(exports, "utf_codepoint_to_int", ()=>(0, _gleamStdlibMjs.utf_codepoint_to_int));
/**
 * Determines if a `String` is empty.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_empty("")
 * ```
 *
 * ```gleam
 * assert !is_empty("the world")
 * ```
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
/**
 * Reverses a `String`.
 *
 * This function has to iterate across the whole `String` so it runs in linear
 * time. Avoid using this in a loop.
 *
 * ## Examples
 *
 * ```gleam
 * assert reverse("stressed") == "desserts"
 * ```
 */ parcelHelpers.export(exports, "reverse", ()=>reverse);
/**
 * Creates a new `String` by replacing all occurrences of a given substring.
 *
 * ## Examples
 *
 * ```gleam
 * assert replace("www.example.com", each: ".", with: "-") == "www-example-com"
 * ```
 *
 * ```gleam
 * assert replace("a,b,c,d,e", each: ",", with: "/") == "a/b/c/d/e"
 * ```
 */ parcelHelpers.export(exports, "replace", ()=>replace);
/**
 * Compares two `String`s to see which is "larger" by comparing their graphemes.
 *
 * This does not compare the size or length of the given `String`s.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/order
 *
 * assert compare("Anthony", "Anthony") == order.Eq
 * ```
 *
 * ```gleam
 * import gleam/order
 *
 * assert compare("A", "B") == order.Lt
 * ```
 */ parcelHelpers.export(exports, "compare", ()=>compare);
/**
 * Takes a substring given a start grapheme index and a length. Negative indexes
 * are taken starting from the *end* of the string.
 *
 * This function runs in linear time with the size of the index and the
 * length. Negative indexes are linear with the size of the input string in
 * addition to the other costs.
 *
 * ## Examples
 *
 * ```gleam
 * assert slice(from: "gleam", at_index: 1, length: 2) == "le"
 * ```
 *
 * ```gleam
 * assert slice(from: "gleam", at_index: 1, length: 10) == "leam"
 * ```
 *
 * ```gleam
 * assert slice(from: "gleam", at_index: 10, length: 3) == ""
 * ```
 *
 * ```gleam
 * assert slice(from: "gleam", at_index: -2, length: 2) == "am"
 * ```
 *
 * ```gleam
 * assert slice(from: "gleam", at_index: -12, length: 2) == ""
 * ```
 */ parcelHelpers.export(exports, "slice", ()=>slice);
/**
 * Drops *n* graphemes from the end of a `String`.
 *
 * This function traverses the full string, so it runs in linear time with the
 * size of the string. Avoid using this in a loop.
 *
 * ## Examples
 *
 * ```gleam
 * assert drop_end(from: "Cigarette Smoking Man", up_to: 2)
 *   == "Cigarette Smoking M"
 * ```
 */ parcelHelpers.export(exports, "drop_end", ()=>drop_end);
/**
 * Creates a new `String` by joining two `String`s together.
 *
 * This function typically copies both `String`s and runs in linear time, but
 * the exact behaviour will depend on how the runtime you are using optimises
 * your code. Benchmark and profile your code if you need to understand its
 * performance better.
 *
 * If you are joining together large string and want to avoid copying any data
 * you may want to investigate using the [`string_tree`](../gleam/string_tree.html)
 * module.
 *
 * ## Examples
 *
 * ```gleam
 * assert append(to: "butter", suffix: "fly") == "butterfly"
 * ```
 */ parcelHelpers.export(exports, "append", ()=>append);
/**
 * Creates a new `String` by joining many `String`s together.
 *
 * This function copies all the `String`s and runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert concat(["never", "the", "less"]) == "nevertheless"
 * ```
 */ parcelHelpers.export(exports, "concat", ()=>concat);
/**
 * Creates a new `String` by repeating a `String` a given number of times.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert repeat("ha", times: 3) == "hahaha"
 * ```
 */ parcelHelpers.export(exports, "repeat", ()=>repeat);
/**
 * Joins many `String`s together with a given separator.
 *
 * This function runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert join(["home","evan","Desktop"], with: "/") == "home/evan/Desktop"
 * ```
 */ parcelHelpers.export(exports, "join", ()=>join);
/**
 * Pads the start of a `String` until it has a given length.
 *
 * ## Examples
 *
 * ```gleam
 * assert pad_start("121", to: 5, with: ".") == "..121"
 * ```
 *
 * ```gleam
 * assert pad_start("121", to: 3, with: ".") == "121"
 * ```
 *
 * ```gleam
 * assert pad_start("121", to: 2, with: ".") == "121"
 * ```
 */ parcelHelpers.export(exports, "pad_start", ()=>pad_start);
/**
 * Pads the end of a `String` until it has a given length.
 *
 * ## Examples
 *
 * ```gleam
 * assert pad_end("123", to: 5, with: ".") == "123.."
 * ```
 *
 * ```gleam
 * assert pad_end("123", to: 3, with: ".") == "123"
 * ```
 *
 * ```gleam
 * assert pad_end("123", to: 2, with: ".") == "123"
 * ```
 */ parcelHelpers.export(exports, "pad_end", ()=>pad_end);
/**
 * Removes whitespace on both sides of a `String`.
 *
 * Whitespace in this function is the set of nonbreakable whitespace
 * codepoints, defined as Pattern_White_Space in [Unicode Standard Annex #31][1].
 *
 * [1]: https://unicode.org/reports/tr31/
 *
 * ## Examples
 *
 * ```gleam
 * assert trim("  hats  \n") == "hats"
 * ```
 */ parcelHelpers.export(exports, "trim", ()=>trim);
/**
 * Creates a list of `String`s by splitting a given string on a given substring.
 *
 * ## Examples
 *
 * ```gleam
 * assert split("home/gleam/desktop/", on: "/")
 *   == ["home", "gleam", "desktop", ""]
 * ```
 */ parcelHelpers.export(exports, "split", ()=>split);
/**
 * Converts a `String` to a `List` of `UtfCodepoint`.
 *
 * See <https://en.wikipedia.org/wiki/Code_point> and
 * <https://en.wikipedia.org/wiki/Unicode#Codespace_and_Code_Points> for an
 * explanation on code points.
 *
 * ## Examples
 *
 * ```gleam
 * assert "a" |> to_utf_codepoints == [UtfCodepoint(97)]
 * ```
 *
 * ```gleam
 * // Semantically the same as:
 * // ["🏳", "️", "‍", "🌈"] or:
 * // [waving_white_flag, variant_selector_16, zero_width_joiner, rainbow]
 * assert "🏳️‍🌈" |> to_utf_codepoints
 *   == [
 *     UtfCodepoint(127987),
 *     UtfCodepoint(65039),
 *     UtfCodepoint(8205),
 *     UtfCodepoint(127752),
 *   ]
 * ```
 */ parcelHelpers.export(exports, "to_utf_codepoints", ()=>to_utf_codepoints);
/**
 * Converts an integer to a `UtfCodepoint`.
 *
 * Returns an `Error` if the integer does not represent a valid UTF codepoint.
 */ parcelHelpers.export(exports, "utf_codepoint", ()=>utf_codepoint);
/**
 * Converts a `String` into `Option(String)` where an empty `String` becomes
 * `None`.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_option("") == None
 * ```
 *
 * ```gleam
 * assert to_option("hats") == Some("hats")
 * ```
 */ parcelHelpers.export(exports, "to_option", ()=>to_option);
/**
 * Returns the first grapheme cluster in a given `String` and wraps it in a
 * `Result(String, Nil)`. If the `String` is empty, it returns `Error(Nil)`.
 * Otherwise, it returns `Ok(String)`.
 *
 * ## Examples
 *
 * ```gleam
 * assert first("") == Error(Nil)
 * ```
 *
 * ```gleam
 * assert first("icecream") == Ok("i")
 * ```
 */ parcelHelpers.export(exports, "first", ()=>first);
/**
 * Returns the last grapheme cluster in a given `String` and wraps it in a
 * `Result(String, Nil)`. If the `String` is empty, it returns `Error(Nil)`.
 * Otherwise, it returns `Ok(String)`.
 *
 * This function traverses the full string, so it runs in linear time with the
 * length of the string. Avoid using this in a loop.
 *
 * ## Examples
 *
 * ```gleam
 * assert last("") == Error(Nil)
 * ```
 *
 * ```gleam
 * assert last("icecream") == Ok("m")
 * ```
 */ parcelHelpers.export(exports, "last", ()=>last);
/**
 * Creates a new `String` with the first grapheme in the input `String`
 * converted to uppercase and the remaining graphemes to lowercase.
 *
 * ## Examples
 *
 * ```gleam
 * assert capitalise("mamouna") == "Mamouna"
 * ```
 */ parcelHelpers.export(exports, "capitalise", ()=>capitalise);
/**
 * Returns a `String` representation of a term in Gleam syntax.
 *
 * This may be occasionally useful for quick-and-dirty printing of values in
 * scripts. For error reporting and other uses prefer constructing strings by
 * pattern matching on the values.
 *
 * ## Limitations
 *
 * The output format of this function is not stable and could change at any
 * time. The output is not suitable for parsing.
 *
 * This function works using runtime reflection, so the output may not be
 * perfectly accurate for data structures where the runtime structure doesn't
 * hold enough information to determine the original syntax. For example,
 * tuples with an Erlang atom in the first position will be mistaken for Gleam
 * records.
 *
 * ## Security and safety
 *
 * There is no limit to how large the strings that this function can produce.
 * Be careful not to call this function with large data structures or you
 * could use very large amounts of memory, potentially causing runtime
 * problems.
 */ parcelHelpers.export(exports, "inspect", ()=>inspect);
/**
 * Drops *n* graphemes from the start of a `String`.
 *
 * This function runs in linear time with the number of graphemes to drop.
 *
 * ## Examples
 *
 * ```gleam
 * assert drop_start(from: "The Lone Gunmen", up_to: 2) == "e Lone Gunmen"
 * ```
 */ parcelHelpers.export(exports, "drop_start", ()=>drop_start);
var _gleamMjs = require("../gleam.mjs");
var _listMjs = require("../gleam/list.mjs");
var _optionMjs = require("../gleam/option.mjs");
var _orderMjs = require("../gleam/order.mjs");
var _stringTreeMjs = require("../gleam/string_tree.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
class Leading extends (0, _gleamMjs.CustomType) {
}
class Trailing extends (0, _gleamMjs.CustomType) {
}
function is_empty(str) {
    return str === "";
}
function reverse(string) {
    let _pipe = string;
    let _pipe$1 = _stringTreeMjs.from_string(_pipe);
    let _pipe$2 = _stringTreeMjs.reverse(_pipe$1);
    return _stringTreeMjs.to_string(_pipe$2);
}
function replace(string, pattern, substitute) {
    let _pipe = string;
    let _pipe$1 = _stringTreeMjs.from_string(_pipe);
    let _pipe$2 = _stringTreeMjs.replace(_pipe$1, pattern, substitute);
    return _stringTreeMjs.to_string(_pipe$2);
}
function compare(a, b) {
    let $ = a === b;
    if ($) return new _orderMjs.Eq();
    else {
        let $1 = (0, _gleamStdlibMjs.less_than)(a, b);
        if ($1) return new _orderMjs.Lt();
        else return new _orderMjs.Gt();
    }
}
function slice(string, idx, len) {
    let $ = len <= 0;
    if ($) return "";
    else {
        let $1 = idx < 0;
        if ($1) {
            let translated_idx = (0, _gleamStdlibMjs.string_length)(string) + idx;
            let $2 = translated_idx < 0;
            if ($2) return "";
            else return (0, _gleamStdlibMjs.string_grapheme_slice)(string, translated_idx, len);
        } else return (0, _gleamStdlibMjs.string_grapheme_slice)(string, idx, len);
    }
}
function drop_end(string, num_graphemes) {
    let $ = num_graphemes <= 0;
    if ($) return string;
    else return slice(string, 0, (0, _gleamStdlibMjs.string_length)(string) - num_graphemes);
}
function append(first, second) {
    return first + second;
}
function concat_loop(loop$strings, loop$accumulator) {
    while(true){
        let strings = loop$strings;
        let accumulator = loop$accumulator;
        if (strings instanceof (0, _gleamMjs.Empty)) return accumulator;
        else {
            let string = strings.head;
            let strings$1 = strings.tail;
            loop$strings = strings$1;
            loop$accumulator = accumulator + string;
        }
    }
}
function concat(strings) {
    return concat_loop(strings, "");
}
function repeat_loop(loop$times, loop$doubling_acc, loop$acc) {
    while(true){
        let times = loop$times;
        let doubling_acc = loop$doubling_acc;
        let acc = loop$acc;
        let _block;
        let $ = times % 2;
        if ($ === 0) _block = acc;
        else _block = acc + doubling_acc;
        let acc$1 = _block;
        let times$1 = globalThis.Math.trunc(times / 2);
        let $1 = times$1 <= 0;
        if ($1) return acc$1;
        else {
            loop$times = times$1;
            loop$doubling_acc = doubling_acc + doubling_acc;
            loop$acc = acc$1;
        }
    }
}
function repeat(string, times) {
    let $ = times <= 0;
    if ($) return "";
    else return repeat_loop(times, string, "");
}
function join_loop(loop$strings, loop$separator, loop$accumulator) {
    while(true){
        let strings = loop$strings;
        let separator = loop$separator;
        let accumulator = loop$accumulator;
        if (strings instanceof (0, _gleamMjs.Empty)) return accumulator;
        else {
            let string = strings.head;
            let strings$1 = strings.tail;
            loop$strings = strings$1;
            loop$separator = separator;
            loop$accumulator = accumulator + separator + string;
        }
    }
}
function join(strings, separator) {
    if (strings instanceof (0, _gleamMjs.Empty)) return "";
    else {
        let first$1 = strings.head;
        let rest = strings.tail;
        return join_loop(rest, separator, first$1);
    }
}
function padding(size, pad_string) {
    let pad_string_length = (0, _gleamStdlibMjs.string_length)(pad_string);
    let num_pads = (0, _gleamMjs.divideInt)(size, pad_string_length);
    let extra = (0, _gleamMjs.remainderInt)(size, pad_string_length);
    return repeat(pad_string, num_pads) + slice(pad_string, 0, extra);
}
function pad_start(string, desired_length, pad_string) {
    let current_length = (0, _gleamStdlibMjs.string_length)(string);
    let to_pad_length = desired_length - current_length;
    let $ = to_pad_length <= 0;
    if ($) return string;
    else return padding(to_pad_length, pad_string) + string;
}
function pad_end(string, desired_length, pad_string) {
    let current_length = (0, _gleamStdlibMjs.string_length)(string);
    let to_pad_length = desired_length - current_length;
    let $ = to_pad_length <= 0;
    if ($) return string;
    else return string + padding(to_pad_length, pad_string);
}
function trim(string) {
    let _pipe = string;
    let _pipe$1 = (0, _gleamStdlibMjs.trim_start)(_pipe);
    return (0, _gleamStdlibMjs.trim_end)(_pipe$1);
}
function to_graphemes_loop(loop$string, loop$acc) {
    while(true){
        let string = loop$string;
        let acc = loop$acc;
        let $ = (0, _gleamStdlibMjs.pop_grapheme)(string);
        if ($ instanceof (0, _gleamMjs.Ok)) {
            let grapheme = $[0][0];
            let rest = $[0][1];
            loop$string = rest;
            loop$acc = (0, _gleamMjs.prepend)(grapheme, acc);
        } else return acc;
    }
}
function split(x, substring) {
    if (substring === "") return (0, _gleamStdlibMjs.graphemes)(x);
    else {
        let _pipe = x;
        let _pipe$1 = _stringTreeMjs.from_string(_pipe);
        let _pipe$2 = _stringTreeMjs.split(_pipe$1, substring);
        return _listMjs.map(_pipe$2, _stringTreeMjs.to_string);
    }
}
function do_to_utf_codepoints(string) {
    let _pipe = string;
    let _pipe$1 = (0, _gleamStdlibMjs.string_to_codepoint_integer_list)(_pipe);
    return _listMjs.map(_pipe$1, (0, _gleamStdlibMjs.codepoint));
}
function to_utf_codepoints(string) {
    return do_to_utf_codepoints(string);
}
function utf_codepoint(value) {
    let i = value;
    if (i > 1114111) return new (0, _gleamMjs.Error)(undefined);
    else {
        let i = value;
        if (i >= 55296 && i <= 57343) return new (0, _gleamMjs.Error)(undefined);
        else {
            let i = value;
            if (i < 0) return new (0, _gleamMjs.Error)(undefined);
            else {
                let i = value;
                return new (0, _gleamMjs.Ok)((0, _gleamStdlibMjs.codepoint)(i));
            }
        }
    }
}
function to_option(string) {
    if (string === "") return new (0, _optionMjs.None)();
    else return new (0, _optionMjs.Some)(string);
}
function first(string) {
    let $ = (0, _gleamStdlibMjs.pop_grapheme)(string);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let first$1 = $[0][0];
        return new (0, _gleamMjs.Ok)(first$1);
    } else return $;
}
function last(string) {
    let $ = (0, _gleamStdlibMjs.pop_grapheme)(string);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let $1 = $[0][1];
        if ($1 === "") {
            let first$1 = $[0][0];
            return new (0, _gleamMjs.Ok)(first$1);
        } else {
            let rest = $1;
            return new (0, _gleamMjs.Ok)(slice(rest, -1, 1));
        }
    } else return $;
}
function capitalise(string) {
    let $ = (0, _gleamStdlibMjs.pop_grapheme)(string);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let first$1 = $[0][0];
        let rest = $[0][1];
        return append((0, _gleamStdlibMjs.uppercase)(first$1), (0, _gleamStdlibMjs.lowercase)(rest));
    } else return "";
}
function inspect(term) {
    let _pipe = term;
    let _pipe$1 = (0, _gleamStdlibMjs.inspect)(_pipe);
    return _stringTreeMjs.to_string(_pipe$1);
}
function drop_start(string, num_graphemes) {
    let $ = num_graphemes <= 0;
    if ($) return string;
    else {
        let prefix = (0, _gleamStdlibMjs.string_grapheme_slice)(string, 0, num_graphemes);
        let prefix_size = (0, _gleamStdlibMjs.byte_size)(prefix);
        return (0, _gleamStdlibMjs.string_byte_slice)(string, prefix_size, (0, _gleamStdlibMjs.byte_size)(string) - prefix_size);
    }
}

},{"../gleam.mjs":"aiPrb","../gleam/list.mjs":"8dUwY","../gleam/option.mjs":"aWtoH","../gleam/order.mjs":"eYj92","../gleam/string_tree.mjs":"8IH0o","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"8IH0o":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "append_tree", ()=>(0, _gleamStdlibMjs.add));
parcelHelpers.export(exports, "byte_size", ()=>(0, _gleamStdlibMjs.length));
parcelHelpers.export(exports, "concat", ()=>(0, _gleamStdlibMjs.concat));
parcelHelpers.export(exports, "from_string", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "from_strings", ()=>(0, _gleamStdlibMjs.concat));
parcelHelpers.export(exports, "lowercase", ()=>(0, _gleamStdlibMjs.lowercase));
parcelHelpers.export(exports, "replace", ()=>(0, _gleamStdlibMjs.string_replace));
parcelHelpers.export(exports, "split", ()=>(0, _gleamStdlibMjs.split));
parcelHelpers.export(exports, "to_string", ()=>(0, _gleamStdlibMjs.identity));
parcelHelpers.export(exports, "uppercase", ()=>(0, _gleamStdlibMjs.uppercase));
/**
 * Prepends some `StringTree` onto the start of another.
 *
 * Runs in constant time.
 */ parcelHelpers.export(exports, "prepend_tree", ()=>prepend_tree);
/**
 * Create an empty `StringTree`. Useful as the start of a pipe chaining many
 * trees together.
 */ parcelHelpers.export(exports, "new$", ()=>new$);
/**
 * Prepends a `String` onto the start of some `StringTree`.
 *
 * Runs in constant time.
 */ parcelHelpers.export(exports, "prepend", ()=>prepend);
/**
 * Appends a `String` onto the end of some `StringTree`.
 *
 * Runs in constant time.
 */ parcelHelpers.export(exports, "append", ()=>append);
/**
 * Joins the given trees into a new tree separated with the given string.
 */ parcelHelpers.export(exports, "join", ()=>join);
/**
 * Converts a `StringTree` to a new one with the contents reversed.
 */ parcelHelpers.export(exports, "reverse", ()=>reverse);
/**
 * Compares two string trees to determine if they have the same textual
 * content.
 *
 * Comparing two string trees using the `==` operator may return `False` even
 * if they have the same content as they may have been build in different ways,
 * so using this function is often preferred.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_strings(["a", "b"]) != from_string("ab")
 * ```
 *
 * ```gleam
 * assert is_equal(from_strings(["a", "b"]), from_string("ab"))
 * ```
 */ parcelHelpers.export(exports, "is_equal", ()=>is_equal);
/**
 * Inspects a `StringTree` to determine if it is equivalent to an empty string.
 *
 * ## Examples
 *
 * ```gleam
 * assert !{ from_string("ok") |> is_empty }
 * ```
 *
 * ```gleam
 * assert from_string("") |> is_empty
 * ```
 *
 * ```gleam
 * assert from_strings([]) |> is_empty
 * ```
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
var _gleamMjs = require("../gleam.mjs");
var _listMjs = require("../gleam/list.mjs");
var _gleamStdlibMjs = require("../gleam_stdlib.mjs");
class All extends (0, _gleamMjs.CustomType) {
}
function prepend_tree(tree, prefix) {
    return (0, _gleamStdlibMjs.add)(prefix, tree);
}
function new$() {
    return (0, _gleamStdlibMjs.concat)((0, _gleamMjs.toList)([]));
}
function prepend(tree, prefix) {
    return (0, _gleamStdlibMjs.add)((0, _gleamStdlibMjs.identity)(prefix), tree);
}
function append(tree, second) {
    return (0, _gleamStdlibMjs.add)(tree, (0, _gleamStdlibMjs.identity)(second));
}
function join(trees, sep) {
    let _pipe = trees;
    let _pipe$1 = _listMjs.intersperse(_pipe, (0, _gleamStdlibMjs.identity)(sep));
    return (0, _gleamStdlibMjs.concat)(_pipe$1);
}
function reverse(tree) {
    let _pipe = tree;
    let _pipe$1 = (0, _gleamStdlibMjs.identity)(_pipe);
    let _pipe$2 = (0, _gleamStdlibMjs.graphemes)(_pipe$1);
    let _pipe$3 = _listMjs.reverse(_pipe$2);
    return (0, _gleamStdlibMjs.concat)(_pipe$3);
}
function is_equal(a, b) {
    return (0, _gleamMjs.isEqual)(a, b);
}
function is_empty(tree) {
    return (0, _gleamMjs.isEqual)((0, _gleamStdlibMjs.identity)(""), tree);
}

},{"../gleam.mjs":"aiPrb","../gleam/list.mjs":"8dUwY","../gleam_stdlib.mjs":"2eNPH","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"9FST8":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "is_browser", ()=>(0, _runtimeFfiMjs.is_browser));
parcelHelpers.export(exports, "is_registered", ()=>(0, _runtimeFfiMjs.is_registered));
parcelHelpers.export(exports, "register", ()=>(0, _componentFfiMjs.make_component));
parcelHelpers.export(exports, "send", ()=>(0, _runtimeFfiMjs.send));
parcelHelpers.export(exports, "start_server_component", ()=>(0, _runtimeFfiMjs1.start));
parcelHelpers.export(exports, "ActorError", ()=>ActorError);
parcelHelpers.export(exports, "Error$ActorError", ()=>Error$ActorError);
parcelHelpers.export(exports, "Error$isActorError", ()=>Error$isActorError);
parcelHelpers.export(exports, "Error$ActorError$reason", ()=>Error$ActorError$reason);
parcelHelpers.export(exports, "Error$ActorError$0", ()=>Error$ActorError$0);
parcelHelpers.export(exports, "BadComponentName", ()=>BadComponentName);
parcelHelpers.export(exports, "Error$BadComponentName", ()=>Error$BadComponentName);
parcelHelpers.export(exports, "Error$isBadComponentName", ()=>Error$isBadComponentName);
parcelHelpers.export(exports, "Error$BadComponentName$name", ()=>Error$BadComponentName$name);
parcelHelpers.export(exports, "Error$BadComponentName$0", ()=>Error$BadComponentName$0);
parcelHelpers.export(exports, "ComponentAlreadyRegistered", ()=>ComponentAlreadyRegistered);
parcelHelpers.export(exports, "Error$ComponentAlreadyRegistered", ()=>Error$ComponentAlreadyRegistered);
parcelHelpers.export(exports, "Error$isComponentAlreadyRegistered", ()=>Error$isComponentAlreadyRegistered);
parcelHelpers.export(exports, "Error$ComponentAlreadyRegistered$name", ()=>Error$ComponentAlreadyRegistered$name);
parcelHelpers.export(exports, "Error$ComponentAlreadyRegistered$0", ()=>Error$ComponentAlreadyRegistered$0);
parcelHelpers.export(exports, "ElementNotFound", ()=>ElementNotFound);
parcelHelpers.export(exports, "Error$ElementNotFound", ()=>Error$ElementNotFound);
parcelHelpers.export(exports, "Error$isElementNotFound", ()=>Error$isElementNotFound);
parcelHelpers.export(exports, "Error$ElementNotFound$selector", ()=>Error$ElementNotFound$selector);
parcelHelpers.export(exports, "Error$ElementNotFound$0", ()=>Error$ElementNotFound$0);
parcelHelpers.export(exports, "NotABrowser", ()=>NotABrowser);
parcelHelpers.export(exports, "Error$NotABrowser", ()=>Error$NotABrowser);
parcelHelpers.export(exports, "Error$isNotABrowser", ()=>Error$isNotABrowser);
/**
 * A complete Lustre application that follows the Model-View-Update architecture
 * and can handle side effects like HTTP requests or querying the DOM. Most real
 * Lustre applications will use this constructor.
 *
 * To learn more about effects and their purpose, take a look at the
 * [`effect`](./lustre/effect.html) module or the
 * [HTTP requests example](https://github.com/lustre-labs/lustre/tree/main/examples/05-http-requests).
 */ parcelHelpers.export(exports, "application", ()=>application);
/**
 * The simplest type of Lustre application. The `element` application is
 * primarily used for demonstration purposes. It renders a static Lustre `Element`
 * on the page and does not have any state or update logic.
 */ parcelHelpers.export(exports, "element", ()=>element);
/**
 * A `simple` application has the basic Model-View-Update building blocks present
 * in all Lustre applications, but it cannot handle effects. This is a great way
 * to learn the basics of Lustre and its architecture.
 *
 * Once you're comfortable with the Model-View-Update loop and want to start
 * building more complex applications that can communicate with the outside world,
 * you'll want to use the [`application`](#application) constructor instead.
 */ parcelHelpers.export(exports, "simple", ()=>simple);
/**
 * A `component` is a type of Lustre application designed to be embedded within
 * another application and has its own encapsulated update loop. This constructor
 * is almost identical to the [`application`](#application) constructor, but it
 * also allows you to specify a dictionary of attribute names and decoders.
 *
 * When a component is rendered in a parent application, it can receive data from
 * the parent application through HTML attributes and properties just like any
 * other HTML element. This dictionary of decoders allows you to specify how to
 * decode those attributes into messages your component's update loop can handle.
 *
 * > **Note**: Lustre components are conceptually a lot "heavier" than components
 * > in frameworks like React. They should be used for more complex UI widgets
 * > like a combobox with complex keyboard interactions rather than simple things
 * > like buttons or text inputs. Where possible try to think about how to build
 * > your UI with simple view functions (functions that return [Elements](./lustre/element.html#Element))
 * > and only reach for components when you really need to encapsulate that update
 * > loop.
 */ parcelHelpers.export(exports, "component", ()=>component);
/**
 * Assign a [`Name`](https://hexdocs.pm/gleam_erlang/gleam/erlang/process.html#Name)
 * to a Lustre application. This is useful for [_supervised_](#supervised) server
 * components as it allows other processes to find and communicate with the
 * runtime even if it is restarted.
 *
 * > **Note**: names must **never** be created dynamically as too many names
 * > will exhaust the atom table and cause the VM to crash. Names should be
 * > created at the start of your program and passed down where needed.
 *
 * > **Note**: a named application should **never** be used to create a
 * > [factory supervisor](#factory) as only one process can be registered under
 * > a given name.
 */ parcelHelpers.export(exports, "named", ()=>named);
/**
 * Create a server component child specification suitable for supervision in a
 * [static supervisor](https://hexdocs.pm/gleam_otp/gleam/otp/static_supervisor.html).
 * This is the preferred way of starting Lustre server components on the Erlang
 * target.
 */ parcelHelpers.export(exports, "supervised", ()=>supervised);
/**
 * Create a [factory supervisor](https://hexdocs.pm/gleam_otp/gleam/otp/factory_supervisor.html)
 * capable of starting many instances of a Lustre server component dynamically.
 * Along with [`supervised`](#supervised), this is one of the ways to ensure
 * proper supervision and fault-tolerance for Lustre server components on the
 * Erlang target.
 */ parcelHelpers.export(exports, "factory", ()=>factory);
/**
 * Build a message for a running application's `update` function.
 *
 * This message can be delivered to the runtime using [`send`](#send), allowing
 * communication with a Lustre app without having to use an effect.
 */ parcelHelpers.export(exports, "dispatch", ()=>dispatch);
/**
 * Instruct a running application to shut down. For client SPAs this will stop
 * the runtime and unmount the app from the DOM. For server components, this will
 * stop the runtime and prevent any further patches from being sent to connected
 * clients.
 */ parcelHelpers.export(exports, "shutdown", ()=>shutdown);
/**
 * Start a constructed application as a client-side single-page application (SPA).
 * This is the most typical way to start a Lustre application and will *only* work
 * in the browser
 *
 * The second argument is a [CSS selector](https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelector)
 * used to locate the DOM element where the application will be mounted on to.
 * The most common selectors are `"#app"` to target an element with an id of `app`
 * or `[data-lustre-app]` to target an element with a `data-lustre-app` attribute.
 *
 * The third argument is the starting data for the application. This is passed
 * to the application's `init` function.
 */ parcelHelpers.export(exports, "start", ()=>start);
var _processMjs = require("../gleam_erlang/gleam/erlang/process.mjs");
var _actorMjs = require("../gleam_otp/gleam/otp/actor.mjs");
var _factorySupervisorMjs = require("../gleam_otp/gleam/otp/factory_supervisor.mjs");
var _supervisionMjs = require("../gleam_otp/gleam/otp/supervision.mjs");
var _boolMjs = require("../gleam_stdlib/gleam/bool.mjs");
var _functionMjs = require("../gleam_stdlib/gleam/function.mjs");
var _optionMjs = require("../gleam_stdlib/gleam/option.mjs");
var _gleamMjs = require("./gleam.mjs");
var _componentMjs = require("./lustre/component.mjs");
var _effectMjs = require("./lustre/effect.mjs");
var _elementMjs = require("./lustre/element.mjs");
var _appMjs = require("./lustre/runtime/app.mjs");
var _componentFfiMjs = require("./lustre/runtime/client/component.ffi.mjs");
var _runtimeFfiMjs = require("./lustre/runtime/client/runtime.ffi.mjs");
var _spaFfiMjs = require("./lustre/runtime/client/spa.ffi.mjs");
var _runtimeFfiMjs1 = require("./lustre/runtime/server/runtime.ffi.mjs");
var _runtimeMjs = require("./lustre/runtime/server/runtime.mjs");
class ActorError extends (0, _gleamMjs.CustomType) {
    constructor(reason){
        super();
        this.reason = reason;
    }
}
const Error$ActorError = (reason)=>new ActorError(reason);
const Error$isActorError = (value)=>value instanceof ActorError;
const Error$ActorError$reason = (value)=>value.reason;
const Error$ActorError$0 = (value)=>value.reason;
class BadComponentName extends (0, _gleamMjs.CustomType) {
    constructor(name){
        super();
        this.name = name;
    }
}
const Error$BadComponentName = (name)=>new BadComponentName(name);
const Error$isBadComponentName = (value)=>value instanceof BadComponentName;
const Error$BadComponentName$name = (value)=>value.name;
const Error$BadComponentName$0 = (value)=>value.name;
class ComponentAlreadyRegistered extends (0, _gleamMjs.CustomType) {
    constructor(name){
        super();
        this.name = name;
    }
}
const Error$ComponentAlreadyRegistered = (name)=>new ComponentAlreadyRegistered(name);
const Error$isComponentAlreadyRegistered = (value)=>value instanceof ComponentAlreadyRegistered;
const Error$ComponentAlreadyRegistered$name = (value)=>value.name;
const Error$ComponentAlreadyRegistered$0 = (value)=>value.name;
class ElementNotFound extends (0, _gleamMjs.CustomType) {
    constructor(selector){
        super();
        this.selector = selector;
    }
}
const Error$ElementNotFound = (selector)=>new ElementNotFound(selector);
const Error$isElementNotFound = (value)=>value instanceof ElementNotFound;
const Error$ElementNotFound$selector = (value)=>value.selector;
const Error$ElementNotFound$0 = (value)=>value.selector;
class NotABrowser extends (0, _gleamMjs.CustomType) {
}
const Error$NotABrowser = ()=>new NotABrowser();
const Error$isNotABrowser = (value)=>value instanceof NotABrowser;
function application(init, update, view) {
    return new (0, _appMjs.App)(new _optionMjs.None(), init, update, view, _appMjs.default_config);
}
function element(view) {
    return application((_)=>{
        return [
            undefined,
            _effectMjs.none()
        ];
    }, (_, _1)=>{
        return [
            undefined,
            _effectMjs.none()
        ];
    }, (_)=>{
        return view;
    });
}
function simple(init, update, view) {
    let init$1 = (arguments$)=>{
        return [
            init(arguments$),
            _effectMjs.none()
        ];
    };
    let update$1 = (model, message)=>{
        return [
            update(model, message),
            _effectMjs.none()
        ];
    };
    return application(init$1, update$1, view);
}
function component(init, update, view, options) {
    return new (0, _appMjs.App)(new _optionMjs.None(), init, update, view, _appMjs.configure(options));
}
function named(app, name) {
    return new (0, _appMjs.App)(new _optionMjs.Some(name), app.init, app.update, app.view, app.config);
}
function supervised(app, arguments$) {
    return _supervisionMjs.worker(()=>{
        return _runtimeMjs.start(app.name, app.init, app.update, app.view, _appMjs.configure_server_component(app.config), arguments$);
    });
}
function factory(app) {
    return _factorySupervisorMjs.worker_child((arguments$)=>{
        return _runtimeMjs.start(app.name, app.init, app.update, app.view, _appMjs.configure_server_component(app.config), arguments$);
    });
}
function dispatch(message) {
    return new _runtimeMjs.EffectDispatchedMessage(message);
}
function shutdown() {
    return new _runtimeMjs.SystemRequestedShutdown();
}
function start(app, selector, arguments$) {
    return _boolMjs.guard(!(0, _runtimeFfiMjs.is_browser)(), new (0, _gleamMjs.Error)(new NotABrowser()), ()=>{
        return (0, _spaFfiMjs.start)(app, selector, arguments$);
    });
}

},{"../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../gleam_otp/gleam/otp/actor.mjs":"jWzax","../gleam_otp/gleam/otp/factory_supervisor.mjs":"fY84x","../gleam_otp/gleam/otp/supervision.mjs":"bzvLg","../gleam_stdlib/gleam/bool.mjs":"5XM1O","../gleam_stdlib/gleam/function.mjs":"2jh6y","../gleam_stdlib/gleam/option.mjs":"aWtoH","./gleam.mjs":"jNPQG","./lustre/component.mjs":"k3Cmy","./lustre/effect.mjs":"iAEPi","./lustre/element.mjs":"2XxJ4","./lustre/runtime/app.mjs":"fnyl8","./lustre/runtime/client/component.ffi.mjs":"eGPg4","./lustre/runtime/client/runtime.ffi.mjs":"eto4y","./lustre/runtime/client/spa.ffi.mjs":"cVaD8","./lustre/runtime/server/runtime.ffi.mjs":"kJVZ5","./lustre/runtime/server/runtime.mjs":"8rUwG","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jb30g":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "ExitMessage", ()=>ExitMessage);
parcelHelpers.export(exports, "ExitMessage$ExitMessage", ()=>ExitMessage$ExitMessage);
parcelHelpers.export(exports, "ExitMessage$isExitMessage", ()=>ExitMessage$isExitMessage);
parcelHelpers.export(exports, "ExitMessage$ExitMessage$pid", ()=>ExitMessage$ExitMessage$pid);
parcelHelpers.export(exports, "ExitMessage$ExitMessage$0", ()=>ExitMessage$ExitMessage$0);
parcelHelpers.export(exports, "ExitMessage$ExitMessage$reason", ()=>ExitMessage$ExitMessage$reason);
parcelHelpers.export(exports, "ExitMessage$ExitMessage$1", ()=>ExitMessage$ExitMessage$1);
parcelHelpers.export(exports, "Normal", ()=>Normal);
parcelHelpers.export(exports, "ExitReason$Normal", ()=>ExitReason$Normal);
parcelHelpers.export(exports, "ExitReason$isNormal", ()=>ExitReason$isNormal);
parcelHelpers.export(exports, "Killed", ()=>Killed);
parcelHelpers.export(exports, "ExitReason$Killed", ()=>ExitReason$Killed);
parcelHelpers.export(exports, "ExitReason$isKilled", ()=>ExitReason$isKilled);
parcelHelpers.export(exports, "Abnormal", ()=>Abnormal);
parcelHelpers.export(exports, "ExitReason$Abnormal", ()=>ExitReason$Abnormal);
parcelHelpers.export(exports, "ExitReason$isAbnormal", ()=>ExitReason$isAbnormal);
parcelHelpers.export(exports, "ExitReason$Abnormal$reason", ()=>ExitReason$Abnormal$reason);
parcelHelpers.export(exports, "ExitReason$Abnormal$0", ()=>ExitReason$Abnormal$0);
parcelHelpers.export(exports, "ProcessDown", ()=>ProcessDown);
parcelHelpers.export(exports, "Down$ProcessDown", ()=>Down$ProcessDown);
parcelHelpers.export(exports, "Down$isProcessDown", ()=>Down$isProcessDown);
parcelHelpers.export(exports, "Down$ProcessDown$monitor", ()=>Down$ProcessDown$monitor);
parcelHelpers.export(exports, "Down$ProcessDown$0", ()=>Down$ProcessDown$0);
parcelHelpers.export(exports, "Down$ProcessDown$pid", ()=>Down$ProcessDown$pid);
parcelHelpers.export(exports, "Down$ProcessDown$1", ()=>Down$ProcessDown$1);
parcelHelpers.export(exports, "Down$ProcessDown$reason", ()=>Down$ProcessDown$reason);
parcelHelpers.export(exports, "Down$ProcessDown$2", ()=>Down$ProcessDown$2);
parcelHelpers.export(exports, "PortDown", ()=>PortDown);
parcelHelpers.export(exports, "Down$PortDown", ()=>Down$PortDown);
parcelHelpers.export(exports, "Down$isPortDown", ()=>Down$isPortDown);
parcelHelpers.export(exports, "Down$PortDown$monitor", ()=>Down$PortDown$monitor);
parcelHelpers.export(exports, "Down$PortDown$0", ()=>Down$PortDown$0);
parcelHelpers.export(exports, "Down$PortDown$port", ()=>Down$PortDown$port);
parcelHelpers.export(exports, "Down$PortDown$1", ()=>Down$PortDown$1);
parcelHelpers.export(exports, "Down$PortDown$reason", ()=>Down$PortDown$reason);
parcelHelpers.export(exports, "Down$PortDown$2", ()=>Down$PortDown$2);
parcelHelpers.export(exports, "Down$monitor", ()=>Down$monitor);
parcelHelpers.export(exports, "Down$reason", ()=>Down$reason);
parcelHelpers.export(exports, "TimerNotFound", ()=>TimerNotFound);
parcelHelpers.export(exports, "Cancelled$TimerNotFound", ()=>Cancelled$TimerNotFound);
parcelHelpers.export(exports, "Cancelled$isTimerNotFound", ()=>Cancelled$isTimerNotFound);
/**
 * The timer was found and cancelled before it triggered.
 *
 * The amount of remaining time before the timer was due to be triggered is
 * returned in milliseconds.
 */ parcelHelpers.export(exports, "Cancelled", ()=>Cancelled);
parcelHelpers.export(exports, "Cancelled$Cancelled", ()=>Cancelled$Cancelled);
parcelHelpers.export(exports, "Cancelled$isCancelled", ()=>Cancelled$isCancelled);
parcelHelpers.export(exports, "Cancelled$Cancelled$time_remaining", ()=>Cancelled$Cancelled$time_remaining);
parcelHelpers.export(exports, "Cancelled$Cancelled$0", ()=>Cancelled$Cancelled$0);
/**
 * Create a subject for the given process with the give tag. This is unsafe!
 * There's nothing here that verifies that the message the subject receives is
 * expected and that the tag is not already in use.
 *
 * You should almost certainly not use this function.
 * 
 * @ignore
 */ parcelHelpers.export(exports, "unsafely_create_subject", ()=>unsafely_create_subject);
/**
 * Create a subject for a name, which can be used to send and receive messages.
 *
 * All subjects created for the same name behave identically and can be used
 * interchangably.
 */ parcelHelpers.export(exports, "named_subject", ()=>named_subject);
/**
 * Get the name of a subject, returning an error if it doesn't have one.
 */ parcelHelpers.export(exports, "subject_name", ()=>subject_name);
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _stringMjs = require("../../../gleam_stdlib/gleam/string.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _atomMjs = require("../../gleam/erlang/atom.mjs");
var _portMjs = require("../../gleam/erlang/port.mjs");
var _referenceMjs = require("../../gleam/erlang/reference.mjs");
class Subject extends (0, _gleamMjs.CustomType) {
    constructor(owner, tag){
        super();
        this.owner = owner;
        this.tag = tag;
    }
}
class NamedSubject extends (0, _gleamMjs.CustomType) {
    constructor(name){
        super();
        this.name = name;
    }
}
class ExitMessage extends (0, _gleamMjs.CustomType) {
    constructor(pid, reason){
        super();
        this.pid = pid;
        this.reason = reason;
    }
}
const ExitMessage$ExitMessage = (pid, reason)=>new ExitMessage(pid, reason);
const ExitMessage$isExitMessage = (value)=>value instanceof ExitMessage;
const ExitMessage$ExitMessage$pid = (value)=>value.pid;
const ExitMessage$ExitMessage$0 = (value)=>value.pid;
const ExitMessage$ExitMessage$reason = (value)=>value.reason;
const ExitMessage$ExitMessage$1 = (value)=>value.reason;
class Normal extends (0, _gleamMjs.CustomType) {
}
const ExitReason$Normal = ()=>new Normal();
const ExitReason$isNormal = (value)=>value instanceof Normal;
class Killed extends (0, _gleamMjs.CustomType) {
}
const ExitReason$Killed = ()=>new Killed();
const ExitReason$isKilled = (value)=>value instanceof Killed;
class Abnormal extends (0, _gleamMjs.CustomType) {
    constructor(reason){
        super();
        this.reason = reason;
    }
}
const ExitReason$Abnormal = (reason)=>new Abnormal(reason);
const ExitReason$isAbnormal = (value)=>value instanceof Abnormal;
const ExitReason$Abnormal$reason = (value)=>value.reason;
const ExitReason$Abnormal$0 = (value)=>value.reason;
class Anything extends (0, _gleamMjs.CustomType) {
}
class Process extends (0, _gleamMjs.CustomType) {
}
class ProcessDown extends (0, _gleamMjs.CustomType) {
    constructor(monitor, pid, reason){
        super();
        this.monitor = monitor;
        this.pid = pid;
        this.reason = reason;
    }
}
const Down$ProcessDown = (monitor, pid, reason)=>new ProcessDown(monitor, pid, reason);
const Down$isProcessDown = (value)=>value instanceof ProcessDown;
const Down$ProcessDown$monitor = (value)=>value.monitor;
const Down$ProcessDown$0 = (value)=>value.monitor;
const Down$ProcessDown$pid = (value)=>value.pid;
const Down$ProcessDown$1 = (value)=>value.pid;
const Down$ProcessDown$reason = (value)=>value.reason;
const Down$ProcessDown$2 = (value)=>value.reason;
class PortDown extends (0, _gleamMjs.CustomType) {
    constructor(monitor, port, reason){
        super();
        this.monitor = monitor;
        this.port = port;
        this.reason = reason;
    }
}
const Down$PortDown = (monitor, port, reason)=>new PortDown(monitor, port, reason);
const Down$isPortDown = (value)=>value instanceof PortDown;
const Down$PortDown$monitor = (value)=>value.monitor;
const Down$PortDown$0 = (value)=>value.monitor;
const Down$PortDown$port = (value)=>value.port;
const Down$PortDown$1 = (value)=>value.port;
const Down$PortDown$reason = (value)=>value.reason;
const Down$PortDown$2 = (value)=>value.reason;
const Down$monitor = (value)=>value.monitor;
const Down$reason = (value)=>value.reason;
class TimerNotFound extends (0, _gleamMjs.CustomType) {
}
const Cancelled$TimerNotFound = ()=>new TimerNotFound();
const Cancelled$isTimerNotFound = (value)=>value instanceof TimerNotFound;
class Cancelled extends (0, _gleamMjs.CustomType) {
    constructor(time_remaining){
        super();
        this.time_remaining = time_remaining;
    }
}
const Cancelled$Cancelled = (time_remaining)=>new Cancelled(time_remaining);
const Cancelled$isCancelled = (value)=>value instanceof Cancelled;
const Cancelled$Cancelled$time_remaining = (value)=>value.time_remaining;
const Cancelled$Cancelled$0 = (value)=>value.time_remaining;
class Kill extends (0, _gleamMjs.CustomType) {
}
function unsafely_create_subject(owner, tag) {
    return new Subject(owner, tag);
}
function named_subject(name) {
    return new NamedSubject(name);
}
function subject_name(subject) {
    if (subject instanceof Subject) return new (0, _gleamMjs.Error)(undefined);
    else {
        let name = subject.name;
        return new (0, _gleamMjs.Ok)(name);
    }
}

},{"../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../gleam.mjs":"3cTd8","../../gleam/erlang/atom.mjs":"h9zDa","../../gleam/erlang/port.mjs":"lteXH","../../gleam/erlang/reference.mjs":"diRwA","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"3cTd8":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"h9zDa":[function(require,module,exports,__globalThis) {
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");

},{"../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7"}],"lteXH":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"diRwA":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jWzax":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Started", ()=>Started);
parcelHelpers.export(exports, "Started$Started", ()=>Started$Started);
parcelHelpers.export(exports, "Started$isStarted", ()=>Started$isStarted);
parcelHelpers.export(exports, "Started$Started$pid", ()=>Started$Started$pid);
parcelHelpers.export(exports, "Started$Started$0", ()=>Started$Started$0);
parcelHelpers.export(exports, "Started$Started$data", ()=>Started$Started$data);
parcelHelpers.export(exports, "Started$Started$1", ()=>Started$Started$1);
parcelHelpers.export(exports, "InitTimeout", ()=>InitTimeout);
parcelHelpers.export(exports, "StartError$InitTimeout", ()=>StartError$InitTimeout);
parcelHelpers.export(exports, "StartError$isInitTimeout", ()=>StartError$isInitTimeout);
parcelHelpers.export(exports, "InitFailed", ()=>InitFailed);
parcelHelpers.export(exports, "StartError$InitFailed", ()=>StartError$InitFailed);
parcelHelpers.export(exports, "StartError$isInitFailed", ()=>StartError$isInitFailed);
parcelHelpers.export(exports, "StartError$InitFailed$0", ()=>StartError$InitFailed$0);
parcelHelpers.export(exports, "InitExited", ()=>InitExited);
parcelHelpers.export(exports, "StartError$InitExited", ()=>StartError$InitExited);
parcelHelpers.export(exports, "StartError$isInitExited", ()=>StartError$isInitExited);
parcelHelpers.export(exports, "StartError$InitExited$0", ()=>StartError$InitExited$0);
/**
 * Indicate the actor should continue, processing any waiting or future messages.
 */ parcelHelpers.export(exports, "continue$", ()=>continue$);
/**
 * Indicate the actor should stop and shut-down, handling no futher messages.
 *
 * The reason for exiting is `Normal`.
 */ parcelHelpers.export(exports, "stop", ()=>stop);
/**
 * Indicate the actor is in a bad state and should shut down. It will not
 * handle any new messages, and any linked processes will also exit abnormally.
 *
 * The provided reason will be given and propagated.
 */ parcelHelpers.export(exports, "stop_abnormal", ()=>stop_abnormal);
/**
 * Provide a selector to change the messages that the actor is handling
 * going forward. This replaces any selector that was previously given
 * in the actor's `init` callback, or in any previous `Next` value.
 */ parcelHelpers.export(exports, "with_selector", ()=>with_selector);
/**
 * Takes the post-initialisation state of the actor. This state will be passed
 * to the `on_message` callback each time a message is received.
 */ parcelHelpers.export(exports, "initialised", ()=>initialised);
/**
 * Add a selector for the actor to receive messages with.
 *
 * If a message is received by the actor but not selected for with the
 * selector then the actor will discard it and log a warning.
 */ parcelHelpers.export(exports, "selecting", ()=>selecting);
/**
 * Add the data to return to the parent process. This might be a subject that
 * the actor will receive messages over.
 */ parcelHelpers.export(exports, "returning", ()=>returning);
/**
 * Create a builder for an actor without a custom initialiser. The actor
 * returns a subject to the parent that can be used to send messages to the
 * actor.
 *
 * If the actor has been given a name with the `named` function then the
 * subject is a named subject.
 *
 * If you wish to create an actor with some other initialisation logic that
 * runs before it starts handling messages, see `new_with_initialiser`.
 */ parcelHelpers.export(exports, "new$", ()=>new$);
/**
 * Create a builder for an actor with a custom initialiser that runs before
 * the start function returns to the parent, and before the actor starts
 * handling messages.
 *
 * The first argument is a number of milliseconds that the initialiser
 * function is expected to return within. If it takes longer the initialiser
 * is considered to have failed and the actor will be killed, and an error
 * will be returned to the parent.
 *
 * The actor's default subject is passed to the initialiser function. You can
 * chose to return it to the parent with `returning`, use it in some other
 * way, or ignore it completely.
 *
 * If a custom selector is given using the `selecting` function then this
 * overwrites the default selector, which selects for the default subject, so
 * you will need to add the subject to the custom selector yourself.
 */ parcelHelpers.export(exports, "new_with_initialiser", ()=>new_with_initialiser);
/**
 * Set the message handler for the actor. This callback function will be
 * called each time the actor receives a message.
 *
 * Actors handle messages sequentially, later messages being handled after the
 * previous one has been handled.
 */ parcelHelpers.export(exports, "on_message", ()=>on_message);
/**
 * Provide a name for the actor to be registered with when started, enabling
 * it to receive messages via a named subject. This is useful for making
 * processes that can take over from an older one that has exited due to a
 * failure, or to avoid passing subjects from receiver processes to sender
 * processes.
 *
 * If the name is already registered to another process then the actor will
 * fail to start.
 *
 * When this function is used the actor's default subject will be a named
 * subject using this name.
 */ parcelHelpers.export(exports, "named", ()=>named);
var _atomMjs = require("../../../gleam_erlang/gleam/erlang/atom.mjs");
var _charlistMjs = require("../../../gleam_erlang/gleam/erlang/charlist.mjs");
var _processMjs = require("../../../gleam_erlang/gleam/erlang/process.mjs");
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _optionMjs = require("../../../gleam_stdlib/gleam/option.mjs");
var _resultMjs = require("../../../gleam_stdlib/gleam/result.mjs");
var _stringMjs = require("../../../gleam_stdlib/gleam/string.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _systemMjs = require("../../gleam/otp/system.mjs");
/**
 * A regular message excepted by the process
 * 
 * @ignore
 */ class Message extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
/**
 * An OTP system message, for debugging or maintenance
 * 
 * @ignore
 */ class System extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
/**
 * An unexpected message
 * 
 * @ignore
 */ class Unexpected extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
/**
 * Continue handling messages.
 *
 * An optional selector can be provided to changes the messages that the
 * actor is handling. This replaces any selector that was previously given
 * in the actor's `init` callback, or in any previous `Next` value.
 * 
 * @ignore
 */ class Continue extends (0, _gleamMjs.CustomType) {
    constructor(state, selector){
        super();
        this.state = state;
        this.selector = selector;
    }
}
/**
 * Stop handling messages and shut down.
 * 
 * @ignore
 */ class Stop extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Self extends (0, _gleamMjs.CustomType) {
    constructor(mode, parent, state, selector, debug_state, message_handler){
        super();
        this.mode = mode;
        this.parent = parent;
        this.state = state;
        this.selector = selector;
        this.debug_state = debug_state;
        this.message_handler = message_handler;
    }
}
class Started extends (0, _gleamMjs.CustomType) {
    constructor(pid, data){
        super();
        this.pid = pid;
        this.data = data;
    }
}
const Started$Started = (pid, data)=>new Started(pid, data);
const Started$isStarted = (value)=>value instanceof Started;
const Started$Started$pid = (value)=>value.pid;
const Started$Started$0 = (value)=>value.pid;
const Started$Started$data = (value)=>value.data;
const Started$Started$1 = (value)=>value.data;
class Initialised extends (0, _gleamMjs.CustomType) {
    constructor(state, selector, return$){
        super();
        this.state = state;
        this.selector = selector;
        this.return = return$;
    }
}
class Builder extends (0, _gleamMjs.CustomType) {
    constructor(initialise, initialisation_timeout, on_message, name){
        super();
        this.initialise = initialise;
        this.initialisation_timeout = initialisation_timeout;
        this.on_message = on_message;
        this.name = name;
    }
}
class InitTimeout extends (0, _gleamMjs.CustomType) {
}
const StartError$InitTimeout = ()=>new InitTimeout();
const StartError$isInitTimeout = (value)=>value instanceof InitTimeout;
class InitFailed extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const StartError$InitFailed = ($0)=>new InitFailed($0);
const StartError$isInitFailed = (value)=>value instanceof InitFailed;
const StartError$InitFailed$0 = (value)=>value[0];
class InitExited extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const StartError$InitExited = ($0)=>new InitExited($0);
const StartError$isInitExited = (value)=>value instanceof InitExited;
const StartError$InitExited$0 = (value)=>value[0];
class Ack extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Mon extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
function continue$(state) {
    return new Continue(state, new (0, _optionMjs.None)());
}
function stop() {
    return new Stop(new _processMjs.Normal());
}
function stop_abnormal(reason) {
    return new Stop(new _processMjs.Abnormal(_dynamicMjs.string(reason)));
}
function with_selector(value, selector) {
    if (value instanceof Continue) {
        let state = value.state;
        return new Continue(state, new (0, _optionMjs.Some)(selector));
    } else return value;
}
function initialised(state) {
    return new Initialised(state, new (0, _optionMjs.None)(), undefined);
}
function selecting(initialised, selector) {
    return new Initialised(initialised.state, new (0, _optionMjs.Some)(selector), initialised.return);
}
function returning(initialised, return$) {
    return new Initialised(initialised.state, initialised.selector, return$);
}
function new$(state) {
    let initialise = (subject)=>{
        let _pipe = initialised(state);
        let _pipe$1 = returning(_pipe, subject);
        return new (0, _gleamMjs.Ok)(_pipe$1);
    };
    return new Builder(initialise, 1000, (state, _)=>{
        return continue$(state);
    }, new _optionMjs.None());
}
function new_with_initialiser(timeout, initialise) {
    return new Builder(initialise, timeout, (state, _)=>{
        return continue$(state);
    }, new _optionMjs.None());
}
function on_message(builder, handler) {
    return new Builder(builder.initialise, builder.initialisation_timeout, handler, builder.name);
}
function named(builder, name) {
    return new Builder(builder.initialise, builder.initialisation_timeout, builder.on_message, new _optionMjs.Some(name));
}

},{"../../../gleam_erlang/gleam/erlang/atom.mjs":"h9zDa","../../../gleam_erlang/gleam/erlang/charlist.mjs":"bGDBH","../../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../../gleam_stdlib/gleam/result.mjs":"oBmFG","../../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../gleam.mjs":"doVDp","../../gleam/otp/system.mjs":"7dk9j","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"bGDBH":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"oBmFG":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Checks whether the result is an `Ok` value.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_ok(Ok(1))
 * ```
 *
 * ```gleam
 * assert !is_ok(Error(Nil))
 * ```
 */ parcelHelpers.export(exports, "is_ok", ()=>is_ok);
/**
 * Checks whether the result is an `Error` value.
 *
 * ## Examples
 *
 * ```gleam
 * assert !is_error(Ok(1))
 * ```
 *
 * ```gleam
 * assert is_error(Error(Nil))
 * ```
 */ parcelHelpers.export(exports, "is_error", ()=>is_error);
/**
 * Updates a value held within the `Ok` of a result by calling a given function
 * on it.
 *
 * If the result is an `Error` rather than `Ok` the function is not called and the
 * result stays the same.
 *
 * ## Examples
 *
 * ```gleam
 * assert map(over: Ok(1), with: fn(x) { x + 1 }) == Ok(2)
 * ```
 *
 * ```gleam
 * assert map(over: Error(1), with: fn(x) { x + 1 }) == Error(1)
 * ```
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Updates a value held within the `Error` of a result by calling a given function
 * on it.
 *
 * If the result is `Ok` rather than `Error` the function is not called and the
 * result stays the same.
 *
 * ## Examples
 *
 * ```gleam
 * assert map_error(over: Error(1), with: fn(x) { x + 1 }) == Error(2)
 * ```
 *
 * ```gleam
 * assert map_error(over: Ok(1), with: fn(x) { x + 1 }) == Ok(1)
 * ```
 */ parcelHelpers.export(exports, "map_error", ()=>map_error);
/**
 * Merges a nested `Result` into a single layer.
 *
 * ## Examples
 *
 * ```gleam
 * assert flatten(Ok(Ok(1))) == Ok(1)
 * ```
 *
 * ```gleam
 * assert flatten(Ok(Error(""))) == Error("")
 * ```
 *
 * ```gleam
 * assert flatten(Error(Nil)) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "flatten", ()=>flatten);
/**
 * "Updates" an `Ok` result by passing its value to a function that yields a result,
 * and returning the yielded result. (This may "replace" the `Ok` with an `Error`.)
 *
 * If the input is an `Error` rather than an `Ok`, the function is not called and
 * the original `Error` is returned.
 *
 * This function is the equivalent of calling `map` followed by `flatten`, and
 * it is useful for chaining together multiple functions that may fail.
 *
 * ## Examples
 *
 * ```gleam
 * assert try(Ok(1), fn(x) { Ok(x + 1) }) == Ok(2)
 * ```
 *
 * ```gleam
 * assert try(Ok(1), fn(x) { Ok(#("a", x)) }) == Ok(#("a", 1))
 * ```
 *
 * ```gleam
 * assert try(Ok(1), fn(_) { Error("Oh no") }) == Error("Oh no")
 * ```
 *
 * ```gleam
 * assert try(Error(Nil), fn(x) { Ok(x + 1) }) == Error(Nil)
 * ```
 */ parcelHelpers.export(exports, "try$", ()=>try$);
/**
 * Extracts the `Ok` value from a result, returning a default value if the result
 * is an `Error`.
 *
 * ## Examples
 *
 * ```gleam
 * assert unwrap(Ok(1), 0) == 1
 * ```
 *
 * ```gleam
 * assert unwrap(Error(""), 0) == 0
 * ```
 */ parcelHelpers.export(exports, "unwrap", ()=>unwrap);
/**
 * Extracts the `Ok` value from a result, evaluating the default function if the result
 * is an `Error`.
 *
 * ## Examples
 *
 * ```gleam
 * assert lazy_unwrap(Ok(1), fn() { 0 }) == 1
 * ```
 *
 * ```gleam
 * assert lazy_unwrap(Error(""), fn() { 0 }) == 0
 * ```
 */ parcelHelpers.export(exports, "lazy_unwrap", ()=>lazy_unwrap);
/**
 * Extracts the `Error` value from a result, returning a default value if the result
 * is an `Ok`.
 *
 * ## Examples
 *
 * ```gleam
 * assert unwrap_error(Error(1), 0) == 1
 * ```
 *
 * ```gleam
 * assert unwrap_error(Ok(""), 0) == 0
 * ```
 */ parcelHelpers.export(exports, "unwrap_error", ()=>unwrap_error);
/**
 * Returns the first value if it is `Ok`, otherwise returns the second value.
 *
 * ## Examples
 *
 * ```gleam
 * assert or(Ok(1), Ok(2)) == Ok(1)
 * ```
 *
 * ```gleam
 * assert or(Ok(1), Error("Error 2")) == Ok(1)
 * ```
 *
 * ```gleam
 * assert or(Error("Error 1"), Ok(2)) == Ok(2)
 * ```
 *
 * ```gleam
 * assert or(Error("Error 1"), Error("Error 2")) == Error("Error 2")
 * ```
 */ parcelHelpers.export(exports, "or", ()=>or);
/**
 * Returns the first value if it is `Ok`, otherwise evaluates the given function for a fallback value.
 *
 * If you need access to the initial error value, use `result.try_recover`.
 *
 * ## Examples
 *
 * ```gleam
 * assert lazy_or(Ok(1), fn() { Ok(2) }) == Ok(1)
 * ```
 *
 * ```gleam
 * assert lazy_or(Ok(1), fn() { Error("Error 2") }) == Ok(1)
 * ```
 *
 * ```gleam
 * assert lazy_or(Error("Error 1"), fn() { Ok(2) }) == Ok(2)
 * ```
 *
 * ```gleam
 * assert lazy_or(Error("Error 1"), fn() { Error("Error 2") })
 *   == Error("Error 2")
 * ```
 */ parcelHelpers.export(exports, "lazy_or", ()=>lazy_or);
/**
 * Combines a list of results into a single result.
 * If all elements in the list are `Ok` then returns an `Ok` holding the list of values.
 * If any element is `Error` then returns the first error.
 *
 * ## Examples
 *
 * ```gleam
 * assert all([Ok(1), Ok(2)]) == Ok([1, 2])
 * ```
 *
 * ```gleam
 * assert all([Ok(1), Error("e")]) == Error("e")
 * ```
 */ parcelHelpers.export(exports, "all", ()=>all);
/**
 * Given a list of results, returns a pair where the first element is a list
 * of all the values inside `Ok` and the second element is a list with all the
 * values inside `Error`. The values in both lists appear in reverse order with
 * respect to their position in the original list of results.
 *
 * ## Examples
 *
 * ```gleam
 * assert partition([Ok(1), Error("a"), Error("b"), Ok(2)])
 *   == #([2, 1], ["b", "a"])
 * ```
 */ parcelHelpers.export(exports, "partition", ()=>partition);
/**
 * Replace the value within a result
 *
 * ## Examples
 *
 * ```gleam
 * assert replace(Ok(1), Nil) == Ok(Nil)
 * ```
 *
 * ```gleam
 * assert replace(Error(1), Nil) == Error(1)
 * ```
 */ parcelHelpers.export(exports, "replace", ()=>replace);
/**
 * Replace the error within a result
 *
 * ## Examples
 *
 * ```gleam
 * assert replace_error(Error(1), Nil) == Error(Nil)
 * ```
 *
 * ```gleam
 * assert replace_error(Ok(1), Nil) == Ok(1)
 * ```
 */ parcelHelpers.export(exports, "replace_error", ()=>replace_error);
/**
 * Given a list of results, returns only the values inside `Ok`.
 *
 * ## Examples
 *
 * ```gleam
 * assert values([Ok(1), Error("a"), Ok(3)]) == [1, 3]
 * ```
 */ parcelHelpers.export(exports, "values", ()=>values);
/**
 * Updates a value held within the `Error` of a result by calling a given function
 * on it, where the given function also returns a result. The two results are
 * then merged together into one result.
 *
 * If the result is an `Ok` rather than `Error` the function is not called and the
 * result stays the same.
 *
 * This function is useful for chaining together computations that may fail
 * and trying to recover from possible errors.
 *
 * If you do not need access to the initial error value, use `result.lazy_or`.
 *
 * ## Examples
 *
 * ```gleam
 * assert Ok(1)
 *   |> try_recover(with: fn(_) { Error("failed to recover") })
 *   == Ok(1)
 * ```
 *
 * ```gleam
 * assert Error(1)
 *   |> try_recover(with: fn(error) { Ok(error + 1) })
 *   == Ok(2)
 * ```
 *
 * ```gleam
 * assert Error(1)
 *   |> try_recover(with: fn(error) { Error("failed to recover") })
 *   == Error("failed to recover")
 * ```
 */ parcelHelpers.export(exports, "try_recover", ()=>try_recover);
var _gleamMjs = require("../gleam.mjs");
var _listMjs = require("../gleam/list.mjs");
function is_ok(result) {
    if (result instanceof (0, _gleamMjs.Ok)) return true;
    else return false;
}
function is_error(result) {
    if (result instanceof (0, _gleamMjs.Ok)) return false;
    else return true;
}
function map(result, fun) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let x = result[0];
        return new (0, _gleamMjs.Ok)(fun(x));
    } else return result;
}
function map_error(result, fun) {
    if (result instanceof (0, _gleamMjs.Ok)) return result;
    else {
        let error = result[0];
        return new (0, _gleamMjs.Error)(fun(error));
    }
}
function flatten(result) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let x = result[0];
        return x;
    } else return result;
}
function try$(result, fun) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let x = result[0];
        return fun(x);
    } else return result;
}
function unwrap(result, default$) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let v = result[0];
        return v;
    } else return default$;
}
function lazy_unwrap(result, default$) {
    if (result instanceof (0, _gleamMjs.Ok)) {
        let v = result[0];
        return v;
    } else return default$();
}
function unwrap_error(result, default$) {
    if (result instanceof (0, _gleamMjs.Ok)) return default$;
    else {
        let e = result[0];
        return e;
    }
}
function or(first, second) {
    if (first instanceof (0, _gleamMjs.Ok)) return first;
    else return second;
}
function lazy_or(first, second) {
    if (first instanceof (0, _gleamMjs.Ok)) return first;
    else return second();
}
function all(results) {
    return _listMjs.try_map(results, (result)=>{
        return result;
    });
}
function partition_loop(loop$results, loop$oks, loop$errors) {
    while(true){
        let results = loop$results;
        let oks = loop$oks;
        let errors = loop$errors;
        if (results instanceof (0, _gleamMjs.Empty)) return [
            oks,
            errors
        ];
        else {
            let $ = results.head;
            if ($ instanceof (0, _gleamMjs.Ok)) {
                let rest = results.tail;
                let a = $[0];
                loop$results = rest;
                loop$oks = (0, _gleamMjs.prepend)(a, oks);
                loop$errors = errors;
            } else {
                let rest = results.tail;
                let e = $[0];
                loop$results = rest;
                loop$oks = oks;
                loop$errors = (0, _gleamMjs.prepend)(e, errors);
            }
        }
    }
}
function partition(results) {
    return partition_loop(results, (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([]));
}
function replace(result, value) {
    if (result instanceof (0, _gleamMjs.Ok)) return new (0, _gleamMjs.Ok)(value);
    else return result;
}
function replace_error(result, error) {
    if (result instanceof (0, _gleamMjs.Ok)) return result;
    else return new (0, _gleamMjs.Error)(error);
}
function values(results) {
    return _listMjs.filter_map(results, (result)=>{
        return result;
    });
}
function try_recover(result, fun) {
    if (result instanceof (0, _gleamMjs.Ok)) return result;
    else {
        let error = result[0];
        return fun(error);
    }
}

},{"../gleam.mjs":"aiPrb","../gleam/list.mjs":"8dUwY","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"doVDp":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"7dk9j":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Running", ()=>Running);
parcelHelpers.export(exports, "Mode$Running", ()=>Mode$Running);
parcelHelpers.export(exports, "Mode$isRunning", ()=>Mode$isRunning);
parcelHelpers.export(exports, "Suspended", ()=>Suspended);
parcelHelpers.export(exports, "Mode$Suspended", ()=>Mode$Suspended);
parcelHelpers.export(exports, "Mode$isSuspended", ()=>Mode$isSuspended);
parcelHelpers.export(exports, "NoDebug", ()=>NoDebug);
parcelHelpers.export(exports, "DebugOption$NoDebug", ()=>DebugOption$NoDebug);
parcelHelpers.export(exports, "DebugOption$isNoDebug", ()=>DebugOption$isNoDebug);
parcelHelpers.export(exports, "StatusInfo", ()=>StatusInfo);
parcelHelpers.export(exports, "StatusInfo$StatusInfo", ()=>StatusInfo$StatusInfo);
parcelHelpers.export(exports, "StatusInfo$isStatusInfo", ()=>StatusInfo$isStatusInfo);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$module", ()=>StatusInfo$StatusInfo$module);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$0", ()=>StatusInfo$StatusInfo$0);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$parent", ()=>StatusInfo$StatusInfo$parent);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$1", ()=>StatusInfo$StatusInfo$1);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$mode", ()=>StatusInfo$StatusInfo$mode);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$2", ()=>StatusInfo$StatusInfo$2);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$debug_state", ()=>StatusInfo$StatusInfo$debug_state);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$3", ()=>StatusInfo$StatusInfo$3);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$state", ()=>StatusInfo$StatusInfo$state);
parcelHelpers.export(exports, "StatusInfo$StatusInfo$4", ()=>StatusInfo$StatusInfo$4);
parcelHelpers.export(exports, "Resume", ()=>Resume);
parcelHelpers.export(exports, "SystemMessage$Resume", ()=>SystemMessage$Resume);
parcelHelpers.export(exports, "SystemMessage$isResume", ()=>SystemMessage$isResume);
parcelHelpers.export(exports, "SystemMessage$Resume$0", ()=>SystemMessage$Resume$0);
parcelHelpers.export(exports, "Suspend", ()=>Suspend);
parcelHelpers.export(exports, "SystemMessage$Suspend", ()=>SystemMessage$Suspend);
parcelHelpers.export(exports, "SystemMessage$isSuspend", ()=>SystemMessage$isSuspend);
parcelHelpers.export(exports, "SystemMessage$Suspend$0", ()=>SystemMessage$Suspend$0);
parcelHelpers.export(exports, "GetState", ()=>GetState);
parcelHelpers.export(exports, "SystemMessage$GetState", ()=>SystemMessage$GetState);
parcelHelpers.export(exports, "SystemMessage$isGetState", ()=>SystemMessage$isGetState);
parcelHelpers.export(exports, "SystemMessage$GetState$0", ()=>SystemMessage$GetState$0);
parcelHelpers.export(exports, "GetStatus", ()=>GetStatus);
parcelHelpers.export(exports, "SystemMessage$GetStatus", ()=>SystemMessage$GetStatus);
parcelHelpers.export(exports, "SystemMessage$isGetStatus", ()=>SystemMessage$isGetStatus);
parcelHelpers.export(exports, "SystemMessage$GetStatus$0", ()=>SystemMessage$GetStatus$0);
var _atomMjs = require("../../../gleam_erlang/gleam/erlang/atom.mjs");
var _processMjs = require("../../../gleam_erlang/gleam/erlang/process.mjs");
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _gleamMjs = require("../../gleam.mjs");
class Running extends (0, _gleamMjs.CustomType) {
}
const Mode$Running = ()=>new Running();
const Mode$isRunning = (value)=>value instanceof Running;
class Suspended extends (0, _gleamMjs.CustomType) {
}
const Mode$Suspended = ()=>new Suspended();
const Mode$isSuspended = (value)=>value instanceof Suspended;
class NoDebug extends (0, _gleamMjs.CustomType) {
}
const DebugOption$NoDebug = ()=>new NoDebug();
const DebugOption$isNoDebug = (value)=>value instanceof NoDebug;
class StatusInfo extends (0, _gleamMjs.CustomType) {
    constructor(module, parent, mode, debug_state, state){
        super();
        this.module = module;
        this.parent = parent;
        this.mode = mode;
        this.debug_state = debug_state;
        this.state = state;
    }
}
const StatusInfo$StatusInfo = (module, parent, mode, debug_state, state)=>new StatusInfo(module, parent, mode, debug_state, state);
const StatusInfo$isStatusInfo = (value)=>value instanceof StatusInfo;
const StatusInfo$StatusInfo$module = (value)=>value.module;
const StatusInfo$StatusInfo$0 = (value)=>value.module;
const StatusInfo$StatusInfo$parent = (value)=>value.parent;
const StatusInfo$StatusInfo$1 = (value)=>value.parent;
const StatusInfo$StatusInfo$mode = (value)=>value.mode;
const StatusInfo$StatusInfo$2 = (value)=>value.mode;
const StatusInfo$StatusInfo$debug_state = (value)=>value.debug_state;
const StatusInfo$StatusInfo$3 = (value)=>value.debug_state;
const StatusInfo$StatusInfo$state = (value)=>value.state;
const StatusInfo$StatusInfo$4 = (value)=>value.state;
class Resume extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const SystemMessage$Resume = ($0)=>new Resume($0);
const SystemMessage$isResume = (value)=>value instanceof Resume;
const SystemMessage$Resume$0 = (value)=>value[0];
class Suspend extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const SystemMessage$Suspend = ($0)=>new Suspend($0);
const SystemMessage$isSuspend = (value)=>value instanceof Suspend;
const SystemMessage$Suspend$0 = (value)=>value[0];
class GetState extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const SystemMessage$GetState = ($0)=>new GetState($0);
const SystemMessage$isGetState = (value)=>value instanceof GetState;
const SystemMessage$GetState$0 = (value)=>value[0];
class GetStatus extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const SystemMessage$GetStatus = ($0)=>new GetStatus($0);
const SystemMessage$isGetStatus = (value)=>value instanceof GetStatus;
const SystemMessage$GetStatus$0 = (value)=>value[0];

},{"../../../gleam_erlang/gleam/erlang/atom.mjs":"h9zDa","../../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../gleam.mjs":"doVDp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"fY84x":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Get a reference to a supervisor using its registered name.
 *
 * If no supervisor has been started using this name then functions
 * using this reference will fail.
 *
 * # Panics
 *
 * Functions using the `Supervisor` reference returned by this function
 * will panic if there is no factory supervisor registered with the name
 * when they are called. Always make sure your supervisors are themselves
 * supervised.
 */ parcelHelpers.export(exports, "get_by_name", ()=>get_by_name);
/**
 * Provide a name for the supervisor to be registered with when started,
 * enabling it be more easily contacted by other processes. This is useful for
 * enabling processes that can take over from an older one that has exited due
 * to a failure.
 *
 * If the name is already registered to another process then the factory
 * supervisor will fail to start.
 */ parcelHelpers.export(exports, "named", ()=>named);
/**
 * To prevent a supervisor from getting into an infinite loop of child
 * process terminations and restarts, a maximum restart tolerance is
 * defined using two integer values specified with keys intensity and
 * period in the above map. Assuming the values MaxR for intensity and MaxT
 * for period, then, if more than MaxR restarts occur within MaxT seconds,
 * the supervisor terminates all child processes and then itself. The
 * termination reason for the supervisor itself in that case will be
 * shutdown. 
 *
 * Intensity defaults to 2 and period defaults to 5.
 */ parcelHelpers.export(exports, "restart_tolerance", ()=>restart_tolerance);
/**
 * Configure the amount of milliseconds a child has to shut down before
 * being brutal killed by the supervisor.
 *
 * If not set the default for a child is 5000ms.
 *
 * This will be ignored if the child is a supervisor itself.
 */ parcelHelpers.export(exports, "timeout", ()=>timeout);
/**
 * Configure the strategy for restarting children when they exit. See the
 * documentation for the `supervision.Restart` for details.
 *
 * If not set the default strategy is `supervision.Transient`, so children
 * will be restarted if they terminate abnormally.
 */ parcelHelpers.export(exports, "restart_strategy", ()=>restart_strategy);
parcelHelpers.export(exports, "init", ()=>init);
parcelHelpers.export(exports, "start_child_callback", ()=>start_child_callback);
/**
 * Configure a supervisor with a child-starting template function.
 *
 * You should use this unless the child processes are also supervisors.
 *
 * The default shutdown timeout is 5000ms. This can be changed with the
 * `timeout` function.
 */ parcelHelpers.export(exports, "worker_child", ()=>worker_child);
/**
 * Configure a supervisor with a template that will start children that are
 * also supervisors.
 *
 * You should only use this if the child processes are also supervisors.
 *
 * Supervisor children have an unlimited amount of time to shutdown, there is
 * no timeout.
 */ parcelHelpers.export(exports, "supervisor_child", ()=>supervisor_child);
var _atomMjs = require("../../../gleam_erlang/gleam/erlang/atom.mjs");
var _processMjs = require("../../../gleam_erlang/gleam/erlang/process.mjs");
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _optionMjs = require("../../../gleam_stdlib/gleam/option.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _actorMjs = require("../../gleam/otp/actor.mjs");
var _result2Mjs = require("../../gleam/otp/internal/result2.mjs");
var _supervisionMjs = require("../../gleam/otp/supervision.mjs");
class Supervisor extends (0, _gleamMjs.CustomType) {
    constructor(pid){
        super();
        this.pid = pid;
    }
}
class NamedSupervisor extends (0, _gleamMjs.CustomType) {
    constructor(name){
        super();
        this.name = name;
    }
}
class Builder extends (0, _gleamMjs.CustomType) {
    constructor(child_type, template, restart_strategy, intensity, period, name){
        super();
        this.child_type = child_type;
        this.template = template;
        this.restart_strategy = restart_strategy;
        this.intensity = intensity;
        this.period = period;
        this.name = name;
    }
}
class Local extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class SimpleOneForOne extends (0, _gleamMjs.CustomType) {
}
class Strategy extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Intensity extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Period extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Id extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Start extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Restart extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Type extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
class Shutdown extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const default_intensity = 2;
const default_period = 5;
const default_restart_strategy = /* @__PURE__ */ new _supervisionMjs.Transient();
function get_by_name(name) {
    return new NamedSupervisor(name);
}
function named(builder, name) {
    return new Builder(builder.child_type, builder.template, builder.restart_strategy, builder.intensity, builder.period, new _optionMjs.Some(name));
}
function restart_tolerance(builder, intensity, period) {
    return new Builder(builder.child_type, builder.template, builder.restart_strategy, intensity, period, builder.name);
}
function timeout(builder, ms) {
    let $ = builder.child_type;
    if ($ instanceof _supervisionMjs.Worker) return new Builder(new _supervisionMjs.Worker(ms), builder.template, builder.restart_strategy, builder.intensity, builder.period, builder.name);
    else return builder;
}
function restart_strategy(builder, restart_strategy) {
    let $ = builder.child_type;
    if ($ instanceof _supervisionMjs.Worker) return new Builder(builder.child_type, builder.template, restart_strategy, builder.intensity, builder.period, builder.name);
    else return builder;
}
function init(start_data) {
    return new (0, _gleamMjs.Ok)(start_data);
}
function start_child_callback(start, argument) {
    let $ = start(argument);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let started = $[0];
        return new _result2Mjs.Ok(started.pid, started.data);
    } else {
        let error = $[0];
        return new _result2Mjs.Error(error);
    }
}
function worker_child(template) {
    return new Builder(new _supervisionMjs.Worker(5000), template, default_restart_strategy, default_intensity, default_period, new _optionMjs.None());
}
function supervisor_child(template) {
    return new Builder(new _supervisionMjs.Supervisor(), template, default_restart_strategy, default_intensity, default_period, new _optionMjs.None());
}

},{"../../../gleam_erlang/gleam/erlang/atom.mjs":"h9zDa","../../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../gleam.mjs":"doVDp","../../gleam/otp/actor.mjs":"jWzax","../../gleam/otp/internal/result2.mjs":"bZMeW","../../gleam/otp/supervision.mjs":"bzvLg","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"bZMeW":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Ok", ()=>Ok);
parcelHelpers.export(exports, "Result2$Ok", ()=>Result2$Ok);
parcelHelpers.export(exports, "Result2$isOk", ()=>Result2$isOk);
parcelHelpers.export(exports, "Result2$Ok$0", ()=>Result2$Ok$0);
parcelHelpers.export(exports, "Result2$Ok$1", ()=>Result2$Ok$1);
parcelHelpers.export(exports, "Error", ()=>Error);
parcelHelpers.export(exports, "Result2$Error", ()=>Result2$Error);
parcelHelpers.export(exports, "Result2$isError", ()=>Result2$isError);
parcelHelpers.export(exports, "Result2$Error$0", ()=>Result2$Error$0);
var _gleamMjs = require("../../../gleam.mjs");
class Ok extends (0, _gleamMjs.CustomType) {
    constructor($0, $1){
        super();
        this[0] = $0;
        this[1] = $1;
    }
}
const Result2$Ok = ($0, $1)=>new Ok($0, $1);
const Result2$isOk = (value)=>value instanceof Ok;
const Result2$Ok$0 = (value)=>value[0];
const Result2$Ok$1 = (value)=>value[1];
class Error extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const Result2$Error = ($0)=>new Error($0);
const Result2$isError = (value)=>value instanceof Error;
const Result2$Error$0 = (value)=>value[0];

},{"../../../gleam.mjs":"doVDp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"bzvLg":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Permanent", ()=>Permanent);
parcelHelpers.export(exports, "Restart$Permanent", ()=>Restart$Permanent);
parcelHelpers.export(exports, "Restart$isPermanent", ()=>Restart$isPermanent);
parcelHelpers.export(exports, "Transient", ()=>Transient);
parcelHelpers.export(exports, "Restart$Transient", ()=>Restart$Transient);
parcelHelpers.export(exports, "Restart$isTransient", ()=>Restart$isTransient);
parcelHelpers.export(exports, "Temporary", ()=>Temporary);
parcelHelpers.export(exports, "Restart$Temporary", ()=>Restart$Temporary);
parcelHelpers.export(exports, "Restart$isTemporary", ()=>Restart$isTemporary);
/**
 * A worker child has to shut-down within a given amount of time.
 */ parcelHelpers.export(exports, "Worker", ()=>Worker);
parcelHelpers.export(exports, "ChildType$Worker", ()=>ChildType$Worker);
parcelHelpers.export(exports, "ChildType$isWorker", ()=>ChildType$isWorker);
parcelHelpers.export(exports, "ChildType$Worker$shutdown_ms", ()=>ChildType$Worker$shutdown_ms);
parcelHelpers.export(exports, "ChildType$Worker$0", ()=>ChildType$Worker$0);
parcelHelpers.export(exports, "Supervisor", ()=>Supervisor);
parcelHelpers.export(exports, "ChildType$Supervisor", ()=>ChildType$Supervisor);
parcelHelpers.export(exports, "ChildType$isSupervisor", ()=>ChildType$isSupervisor);
parcelHelpers.export(exports, "ChildSpecification", ()=>ChildSpecification);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification", ()=>ChildSpecification$ChildSpecification);
parcelHelpers.export(exports, "ChildSpecification$isChildSpecification", ()=>ChildSpecification$isChildSpecification);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$start", ()=>ChildSpecification$ChildSpecification$start);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$0", ()=>ChildSpecification$ChildSpecification$0);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$restart", ()=>ChildSpecification$ChildSpecification$restart);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$1", ()=>ChildSpecification$ChildSpecification$1);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$significant", ()=>ChildSpecification$ChildSpecification$significant);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$2", ()=>ChildSpecification$ChildSpecification$2);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$child_type", ()=>ChildSpecification$ChildSpecification$child_type);
parcelHelpers.export(exports, "ChildSpecification$ChildSpecification$3", ()=>ChildSpecification$ChildSpecification$3);
/**
 * A regular child process.
 *
 * You should use this unless your process is also a supervisor.
 *
 * The default shutdown timeout is 5000ms. This can be changed with the
 * `timeout` function.
 */ parcelHelpers.export(exports, "worker", ()=>worker);
/**
 * A special child that is a supervisor itself.
 *
 * Supervisor children have an unlimited shutdown time, there is no timeout.
 */ parcelHelpers.export(exports, "supervisor", ()=>supervisor);
/**
 * This defines if a child is considered significant for automatic
 * self-shutdown of the supervisor.
 *
 * You most likely do not want to consider any children significant.
 *
 * This will be ignored if the supervisor auto shutdown is set to `Never`,
 * which is the default.
 *
 * The default value for significance is `False`.
 */ parcelHelpers.export(exports, "significant", ()=>significant);
/**
 * This defines the amount of milliseconds a child has to shut down before
 * being brutal killed by the supervisor.
 *
 * If not set the default for a child is 5000ms.
 *
 * This will be ignored if the child is a supervisor itself.
 */ parcelHelpers.export(exports, "timeout", ()=>timeout);
/**
 * When the child is to be restarted. See the `Restart` documentation for
 * more.
 *
 * The default value for restart is `Permanent`.
 */ parcelHelpers.export(exports, "restart", ()=>restart);
/**
 * Transform the data of the started child process.
 */ parcelHelpers.export(exports, "map_data", ()=>map_data);
var _gleamMjs = require("../../gleam.mjs");
var _actorMjs = require("../../gleam/otp/actor.mjs");
class Permanent extends (0, _gleamMjs.CustomType) {
}
const Restart$Permanent = ()=>new Permanent();
const Restart$isPermanent = (value)=>value instanceof Permanent;
class Transient extends (0, _gleamMjs.CustomType) {
}
const Restart$Transient = ()=>new Transient();
const Restart$isTransient = (value)=>value instanceof Transient;
class Temporary extends (0, _gleamMjs.CustomType) {
}
const Restart$Temporary = ()=>new Temporary();
const Restart$isTemporary = (value)=>value instanceof Temporary;
class Worker extends (0, _gleamMjs.CustomType) {
    constructor(shutdown_ms){
        super();
        this.shutdown_ms = shutdown_ms;
    }
}
const ChildType$Worker = (shutdown_ms)=>new Worker(shutdown_ms);
const ChildType$isWorker = (value)=>value instanceof Worker;
const ChildType$Worker$shutdown_ms = (value)=>value.shutdown_ms;
const ChildType$Worker$0 = (value)=>value.shutdown_ms;
class Supervisor extends (0, _gleamMjs.CustomType) {
}
const ChildType$Supervisor = ()=>new Supervisor();
const ChildType$isSupervisor = (value)=>value instanceof Supervisor;
class ChildSpecification extends (0, _gleamMjs.CustomType) {
    constructor(start, restart, significant, child_type){
        super();
        this.start = start;
        this.restart = restart;
        this.significant = significant;
        this.child_type = child_type;
    }
}
const ChildSpecification$ChildSpecification = (start, restart, significant, child_type)=>new ChildSpecification(start, restart, significant, child_type);
const ChildSpecification$isChildSpecification = (value)=>value instanceof ChildSpecification;
const ChildSpecification$ChildSpecification$start = (value)=>value.start;
const ChildSpecification$ChildSpecification$0 = (value)=>value.start;
const ChildSpecification$ChildSpecification$restart = (value)=>value.restart;
const ChildSpecification$ChildSpecification$1 = (value)=>value.restart;
const ChildSpecification$ChildSpecification$significant = (value)=>value.significant;
const ChildSpecification$ChildSpecification$2 = (value)=>value.significant;
const ChildSpecification$ChildSpecification$child_type = (value)=>value.child_type;
const ChildSpecification$ChildSpecification$3 = (value)=>value.child_type;
function worker(start) {
    return new ChildSpecification(start, new Permanent(), false, new Worker(5000));
}
function supervisor(start) {
    return new ChildSpecification(start, new Permanent(), false, new Supervisor());
}
function significant(child, significant) {
    return new ChildSpecification(child.start, child.restart, significant, child.child_type);
}
function timeout(child, ms) {
    let $ = child.child_type;
    if ($ instanceof Worker) return new ChildSpecification(child.start, child.restart, child.significant, new Worker(ms));
    else return child;
}
function restart(child, restart) {
    return new ChildSpecification(child.start, restart, child.significant, child.child_type);
}
function map_data(child, transform) {
    return new ChildSpecification(()=>{
        let $ = child.start();
        if ($ instanceof (0, _gleamMjs.Ok)) {
            let started = $[0];
            return new (0, _gleamMjs.Ok)(new _actorMjs.Started(started.pid, transform(started.data)));
        } else return $;
    }, child.restart, child.significant, child.child_type);
}

},{"../../gleam.mjs":"doVDp","../../gleam/otp/actor.mjs":"jWzax","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5XM1O":[function(require,module,exports,__globalThis) {
/**
 * Returns the and of two bools, but it evaluates both arguments.
 *
 * It's the function equivalent of the `&&` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert and(True, True)
 * ```
 *
 * ```gleam
 * assert !and(False, True)
 * ```
 *
 * ```gleam
 * assert !and(False, True)
 * ```
 *
 * ```gleam
 * assert !and(False, False)
 * ```
 */ var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "and", ()=>and);
/**
 * Returns the or of two bools, but it evaluates both arguments.
 *
 * It's the function equivalent of the `||` operator.
 * This function is useful in higher order functions or pipes.
 *
 * ## Examples
 *
 * ```gleam
 * assert or(True, True)
 * ```
 *
 * ```gleam
 * assert or(False, True)
 * ```
 *
 * ```gleam
 * assert or(True, False)
 * ```
 *
 * ```gleam
 * assert !or(False, False)
 * ```
 */ parcelHelpers.export(exports, "or", ()=>or);
/**
 * Returns the opposite bool value.
 *
 * This is the same as the `!` or `not` operators in some other languages.
 *
 * ## Examples
 *
 * ```gleam
 * assert !negate(True)
 * ```
 *
 * ```gleam
 * assert negate(False)
 * ```
 */ parcelHelpers.export(exports, "negate", ()=>negate);
/**
 * Returns the nor of two bools.
 *
 * ## Examples
 *
 * ```gleam
 * assert nor(False, False)
 * ```
 *
 * ```gleam
 * assert !nor(False, True)
 * ```
 *
 * ```gleam
 * assert !nor(True, False)
 * ```
 *
 * ```gleam
 * assert !nor(True, True)
 * ```
 */ parcelHelpers.export(exports, "nor", ()=>nor);
/**
 * Returns the nand of two bools.
 *
 * ## Examples
 *
 * ```gleam
 * assert nand(False, False)
 * ```
 *
 * ```gleam
 * assert nand(False, True)
 * ```
 *
 * ```gleam
 * assert nand(True, False)
 * ```
 *
 * ```gleam
 * assert !nand(True, True)
 * ```
 */ parcelHelpers.export(exports, "nand", ()=>nand);
/**
 * Returns the exclusive or of two bools.
 *
 * ## Examples
 *
 * ```gleam
 * assert !exclusive_or(False, False)
 * ```
 *
 * ```gleam
 * assert exclusive_or(False, True)
 * ```
 *
 * ```gleam
 * assert exclusive_or(True, False)
 * ```
 *
 * ```gleam
 * assert !exclusive_or(True, True)
 * ```
 */ parcelHelpers.export(exports, "exclusive_or", ()=>exclusive_or);
/**
 * Returns the exclusive nor of two bools.
 *
 * ## Examples
 *
 * ```gleam
 * assert exclusive_nor(False, False)
 * ```
 *
 * ```gleam
 * assert !exclusive_nor(False, True)
 * ```
 *
 * ```gleam
 * assert !exclusive_nor(True, False)
 * ```
 *
 * ```gleam
 * assert exclusive_nor(True, True)
 * ```
 */ parcelHelpers.export(exports, "exclusive_nor", ()=>exclusive_nor);
/**
 * Returns a string representation of the given bool.
 *
 * ## Examples
 *
 * ```gleam
 * assert to_string(True) == "True"
 * ```
 *
 * ```gleam
 * assert to_string(False) == "False"
 * ```
 */ parcelHelpers.export(exports, "to_string", ()=>to_string);
/**
 * Run a callback function if the given bool is `False`, otherwise return a
 * default value.
 *
 * With a `use` expression this function can simulate the early-return pattern
 * found in some other programming languages.
 *
 * In a procedural language:
 *
 * ```js
 * if (predicate) return value;
 * // ...
 * ```
 *
 * In Gleam with a `use` expression:
 *
 * ```gleam
 * use <- guard(when: predicate, return: value)
 * // ...
 * ```
 *
 * Like everything in Gleam `use` is an expression, so it short circuits the
 * current block, not the entire function. As a result you can assign the value
 * to a variable:
 *
 * ```gleam
 * let x = {
 *   use <- guard(when: predicate, return: value)
 *   // ...
 * }
 * ```
 *
 * Note that unlike in procedural languages the `return` value is evaluated
 * even when the predicate is `False`, so it is advisable not to perform
 * expensive computation nor side-effects there.
 *
 *
 * ## Examples
 *
 * ```gleam
 * let name = ""
 * use <- guard(when: name == "", return: "Welcome!")
 * "Hello, " <> name
 * // -> "Welcome!"
 * ```
 *
 * ```gleam
 * let name = "Kamaka"
 * use <- guard(when: name == "", return: "Welcome!")
 * "Hello, " <> name
 * // -> "Hello, Kamaka"
 * ```
 */ parcelHelpers.export(exports, "guard", ()=>guard);
/**
 * Runs a callback function if the given bool is `True`, otherwise runs an
 * alternative callback function.
 *
 * Useful when further computation should be delayed regardless of the given
 * bool's value.
 *
 * See [`guard`](#guard) for more info.
 *
 * ## Examples
 *
 * ```gleam
 * let name = "Kamaka"
 * let inquiry = fn() { "How may we address you?" }
 * use <- lazy_guard(when: name == "", return: inquiry)
 * "Hello, " <> name
 * // -> "Hello, Kamaka"
 * ```
 *
 * ```gleam
 * import gleam/int
 *
 * let name = ""
 * let greeting = fn() { "Hello, " <> name }
 * use <- lazy_guard(when: name == "", otherwise: greeting)
 * let number = int.random(99)
 * let name = "User " <> int.to_string(number)
 * "Welcome, " <> name
 * // -> "Welcome, User 54"
 * ```
 */ parcelHelpers.export(exports, "lazy_guard", ()=>lazy_guard);
function and(a, b) {
    return a && b;
}
function or(a, b) {
    return a || b;
}
function negate(bool) {
    return !bool;
}
function nor(a, b) {
    return !(a || b);
}
function nand(a, b) {
    return !(a && b);
}
function exclusive_or(a, b) {
    return a !== b;
}
function exclusive_nor(a, b) {
    return a === b;
}
function to_string(bool) {
    if (bool) return "True";
    else return "False";
}
function guard(requirement, consequence, alternative) {
    if (requirement) return consequence;
    else return alternative();
}
function lazy_guard(requirement, consequence, alternative) {
    if (requirement) return consequence();
    else return alternative();
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"2jh6y":[function(require,module,exports,__globalThis) {
/**
 * Takes a single argument and always returns its input value.
 */ var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "identity", ()=>identity);
function identity(x) {
    return x;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jNPQG":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"k3Cmy":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Register a decoder to run whenever the named attribute changes. Attributes
 * can be set in Lustre using the [`attribute`](./attribute.html#attribute)
 * function, set directly on the component's HTML tag, or in JavaScript using
 * the [`setAttribute`](https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute)
 * method.
 *
 * Attributes are always strings, but your decoder is responsible for decoding
 * the string into a message that your component can understand.
 */ parcelHelpers.export(exports, "on_attribute_change", ()=>on_attribute_change);
/**
 * Register decoder to run whenever the given property is set on the component.
 * Properties can be set in Lustre using the [`property`](./attribute.html#property)
 * function or in JavaScript by setting a property directly on the component
 * object.
 *
 * Properties can be any JavaScript object. For server components, properties
 * will be any _JSON-serialisable_ value.
 */ parcelHelpers.export(exports, "on_property_change", ()=>on_property_change);
/**
 * Register a decoder to run whenever a parent component or application
 * [provides](./effect.html#provide) a new context value for the given `key`.
 * Contexts are a powerful feature that allow parents to inject data into
 * child components without knowledge of the DOM structurre, making them great
 * for advanced use-cases like design systems and flexible component hierarchies.
 *
 * Contexts can be any JavaScript object. For server components, contexts will
 * be any _JSON-serialisable_ value.
 */ parcelHelpers.export(exports, "on_context_change", ()=>on_context_change);
/**
 * Mark a component as "form-associated". This lets your component participate
 * in form submission and respond to additional form-specific events such as
 * the form being reset or the browser autofilling this component's value.
 *
 * > **Note**: form-associated components are not supported in server components
 * > for both technical and ideological reasons. If you'd like a component that
 * > participates in form submission, you should use a client component!
 */ parcelHelpers.export(exports, "form_associated", ()=>form_associated);
/**
 * Register a callback that runs when the browser autofills this
 * [form-associated](#form_associated) component's `"value"` attribute. The
 * callback should convert the autofilled value into a message that you handle
 * in your `update` function.
 *
 * > **Note**: server components cannot participate in form submission and configuring
 * > this option will do nothing.
 */ parcelHelpers.export(exports, "on_form_autofill", ()=>on_form_autofill);
/**
 * Set a message to be dispatched whenever a form containing this
 * [form-associated](#form_associated) component is reset.
 *
 * > **Note**: server components cannot participate in form submission and configuring
 * > this option will do nothing.
 */ parcelHelpers.export(exports, "on_form_reset", ()=>on_form_reset);
/**
 * Set a callback that runs when the browser restores this
 * [form-associated](#form_associated) component's `"value"` attribute. This is
 * often triggered when the user navigates back or forward in their history.
 *
 * > **Note**: server components cannot participate in form submission and configuring
 * > this option will do nothing.
 */ parcelHelpers.export(exports, "on_form_restore", ()=>on_form_restore);
/**
 * Configure whether a component's [Shadow Root](https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot)
 * is open or closed. A closed shadow root means the elements rendered inside
 * the component are not accessible from JavaScript outside the component.
 *
 * By default a component's shadow root is **open**. You may want to configure
 * this option manually if you intend to build a component for use outside of
 * Lustre.
 */ parcelHelpers.export(exports, "open_shadow_root", ()=>open_shadow_root);
/**
 * Configure whether a component should attempt to adopt stylesheets from
 * its parent document. Components in Lustre use the shadow DOM to unlock native
 * web component features like slots, but this means elements rendered inside a
 * component are isolated from the document's styles.
 *
 * To get around this, Lustre can attempt to adopt all stylesheets from the
 * parent document when the component is first created; meaning in many cases
 * you can use the same CSS to style your components as you do the rest of your
 * application.
 *
 * By default, this option is **enabled**. You may want to disable this option
 * if you are building a component for use outside of Lustre and do not want
 * document styles to interfere with your component's styling
 */ parcelHelpers.export(exports, "adopt_styles", ()=>adopt_styles);
/**
 * Indicates whether or not this component should delegate focus to its children.
 * When set to `True`, a number of focus-related features are enabled:
 *
 * - Clicking on any non-interactive part of the component will automatically
 *   focus the first focusable child element.
 *
 * - The component can receive focus through the `.focus()` method or the
 *   `autofocus` attribute, and it will automatically focus the first
 *   focusable child element.
 *
 * - The component receives the `:focus` CSS pseudo-class when any of its
 *   focusable children have focus.
 *
 * By default this option is **disabled**. You may want to enable this option
 * when creating complex interactive widgets.
 */ parcelHelpers.export(exports, "delegates_focus", ()=>delegates_focus);
/**
 * Set a message to be sent when a client component is connected to a document
 * or a server component registers a new connection.
 *
 * ## Client components
 *
 * The provided message will be dispatched when the component is connected to a
 * new document. This corresponds to the custom element `connectedCallback` and
 * is a good signal to perform effects that interact with the DOM or many browser
 * APIs.
 *
 * ## Server components
 *
 * The provided message will be dispatched when a new connection is registered
 * by either [`server_component.register_subject`](./server_component.html#register_subject)
 * or [`server_component.register_callback`](./server_component.html#register_callback).
 * Importantly, repeated calls to either of these functions will **not** trigger
 * the message multiple times.
 */ parcelHelpers.export(exports, "on_connect", ()=>on_connect);
/**
 * The message provided to this option will be dispatched whenever a client component
 * is adopted into a new document.
 *
 * > **Note**: this option is only useful for components that will be built and
 * > distributed outside of a typical Lustre application.
 */ parcelHelpers.export(exports, "on_adopt", ()=>on_adopt);
/**
 * Set a message to be sent when a client component is disconnected from a document
 * or a server component deregisters a connection.
 *
 * ## Client components
 *
 * The provided message will be dispatched when the component is disconnected from
 * a document, for example when the element is no longer rendered by your app's
 * `view` function. This corresponds to the custom element `disconnectedCallback`
 * and should be used to clean up any effects.
 *
 * ## Server components
 *
 * The provided message will be dispatched when a connection is deregistered by
 * either [`server_component.deregister_subject`](./server_component.html#deregister_subject)
 * or [`server_component.deregister_callback`](./server_component.html#deregister_callback).
 */ parcelHelpers.export(exports, "on_disconnect", ()=>on_disconnect);
/**
 * Create a default slot for a component. Any elements rendered as children of
 * the component will be placed inside the default slot unless explicitly
 * redirected using the [`slot`](#slot) attribute.
 *
 * If no children are placed into the slot, the `fallback` elements will be
 * rendered instead.
 *
 * To learn more about Shadow DOM and slots, see this excellent guide:
 *
 *   - https://javascript.info/slots-composition
 */ parcelHelpers.export(exports, "default_slot", ()=>default_slot);
/**
 * Create a named slot for a component. Any elements rendered as children of
 * the component with a [`slot`](#slot) attribute matching the `name` will be
 * rendered inside this slot.
 *
 * If no children are placed into the slot, the `fallback` elements will be
 * rendered instead.
 *
 * To learn more about Shadow DOM and slots, see this excellent guide:
 *
 *   - https://javascript.info/slots-composition
 */ parcelHelpers.export(exports, "named_slot", ()=>named_slot);
/**
 * Lustre's component system is built on top the Custom Elements API and the
 * Shadow DOM API. A component's `view` function is rendered inside a shadow
 * root, which means the component's HTML is isolated from the rest of the
 * document.
 *
 * This can make it difficult to style components from CSS outside the component.
 * To help with this, the `part` attribute lets you expose parts of your component
 * by name to be styled by external CSS.
 *
 * For example, if the `view` function for a component called `"my-component`"
 * looks like this:
 *
 * ```gleam
 * import gleam/int
 * import lustre/component
 * import lustre/element/html
 *
 * fn view(model) {
 *   html.div([], [
 *     html.button([], [html.text("-")]),
 *     html.p([component.part("count")], [html.text(int.to_string(model.count))]),
 *     html.button([], [html.text("+")]),
 *   ])
 * }
 * ```
 *
 * Then the following CSS in the **parent** document can be used to style the
 * `<p>` element:
 *
 * ```css
 * my-component::part(count) {
 *   color: red;
 * }
 * ```
 *
 * To learn more about the CSS Shadow Parts specification, see:
 *
 *   - https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/part
 *
 *   - https://developer.mozilla.org/en-US/docs/Web/CSS/::part
 */ parcelHelpers.export(exports, "part", ()=>part);
/**
 * A convenience function that makes it possible to toggle different parts on or
 * off in a single call. This is useful for example when you have a menu item
 * that may be active and you want to conditionally assign the `"active"` part:
 *
 * ```gleam
 * import lustre/component
 * import lustre/element/html
 *
 * fn view(item) {
 *   html.li(
 *     [
 *       component.parts([
 *         #("item", True)
 *         #("active", item.is_active)
 *       ]),
 *     ],
 *     [html.text(item.label)],
 *   ])
 * }
 * ```
 */ parcelHelpers.export(exports, "parts", ()=>parts);
/**
 * While the [`part`](#part) attribute can be used to expose parts of a component
 * to its parent, these parts will not automatically become available to the
 * _document_ when components are nested inside each other.
 *
 * The `exportparts` attribute lets you forward the parts of a nested component
 * to the parent component so they can be styled from the parent document.
 *
 * Consider we have two components, `"my-component"` and `"my-nested-component"`
 * with the following `view` functions:
 *
 * ```gleam
 * import gleam/int
 * import lustre/attribute.{property}
 * import lustre/component
 * import lustre/element.{element}
 * import lustre/element/html
 *
 * fn my_component_view(model) {
 *   html.div([], [
 *     html.button([], [html.text("-")]),
 *     element(
 *       "my-nested-component",
 *       [
 *         property("count", model.count),
 *         component.exportparts(["count"]),
 *       ],
 *       []
 *     )
 *     html.button([], [html.text("+")]),
 *   ])
 * }
 *
 * fn my_nested_component_view(model) {
 *   html.p([component.part("count")], [html.text(int.to_string(model.count))])
 * }
 * ```
 *
 * The `<my-nested-component />` component has a part called `"count"` which the
 * `<my-component />` then forwards to the parent document using the `"exportparts"`
 * attribute. Now the following CSS can be used to style the `<p>` element nested
 * deep inside the `<my-component />`:
 *
 * ```css
 * my-component::part(count) {
 *   color: red;
 * }
 * ```
 *
 * Notice how the styles are applied to the `<my-component />` element, not the
 * `<my-nested-component />` element!
 *
 * To learn more about the CSS Shadow Parts specification, see:
 *
 *   - https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/exportparts
 *
 *   - https://developer.mozilla.org/en-US/docs/Web/CSS/::part
 */ parcelHelpers.export(exports, "exportparts", ()=>exportparts);
/**
 * Associate an element with a [named slot](#named_slot) in a component. Multiple
 * elements can be associated with the same slot name.
 *
 * To learn more about Shadow DOM and slots, see:
 *
 *   - https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/slot
 *
 *   - https://javascript.info/slots-composition
 */ parcelHelpers.export(exports, "slot", ()=>slot);
/**
 * Set the value of a [form-associated component](#form_associated). If the
 * component is rendered inside a `<form>` element, the value will be
 * automatically included in the form submission and available in the form's
 * `FormData` object.
 */ parcelHelpers.export(exports, "set_form_value", ()=>set_form_value);
/**
 * Clear a form value previously set with [`set_form_value`](#set_form_value).
 * When the form is submitted, this component's value will not be included in
 * the form data.
 */ parcelHelpers.export(exports, "clear_form_value", ()=>clear_form_value);
/**
 * Set a custom state on the component. This state is not reflected in the DOM
 * but can be selected in CSS using the `:state` pseudo-class. For example,
 * calling `set_pseudo_state("checked")` on a component called `"my-checkbox"`
 * means the following CSS will apply:
 *
 * ```css
 * my-checkbox:state(checked) {
 *   border: solid;
 * }
 * ```
 *
 * If you are styling a component by rendering a `<style>` element _inside_ the
 * component, the previous CSS would be rewritten as:
 *
 * ```css
 * :host(:state(checked)) {
 *   border: solid;
 * }
 * ```
 */ parcelHelpers.export(exports, "set_pseudo_state", ()=>set_pseudo_state);
/**
 * Remove a custom state set by [`set_pseudo_state`](#set_pseudo_state).
 */ parcelHelpers.export(exports, "remove_pseudo_state", ()=>remove_pseudo_state);
/**
 * Prerender a component with a declarative shadow DOM. This is different to
 * just rendering the component's tag because it also renders the component's
 * internal `view`. Calling this when server-rendering a component allows components
 * to benefit from hydration by providing an initial HTML structure similar to
 * hydratation for client applications.
 *
 * If the component responds to attribute changes, the attributes passed here
 * will be applied before the component is rendered.
 *
 * To support both prerendering and client-side rendering, component authors
 * can use [`lustre.is_browser`](../lustre.html#is_browser) to detect the
 * environment and prerender the component where appropriate:
 *
 * ```gleam
 * import lustre.{type App}
 * import lustre/attribute.{type Attribute}
 * import lustre/component
 * import lustre/element.{type Element, element}
 *
 * pub fn element(
 *   attributes: List(Attribute(message)),
 *   children: List(Element(message))
 * ) -> Element(message) {
 *   case lustre.is_browser() {
 *     True -> element(tag, attributes, children)
 *     False -> component.prerender(component(), tag, attributes, children)
 *   }
 * }
 *
 * const tag = "my-component"
 *
 * fn component() -> App(Nil, Model, Message) {
 *   lustre.component(init:, update:, view:, options:)
 * }
 * ```
 */ parcelHelpers.export(exports, "prerender", ()=>prerender);
var _dynamicMjs = require("../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _listMjs = require("../../gleam_stdlib/gleam/list.mjs");
var _optionMjs = require("../../gleam_stdlib/gleam/option.mjs");
var _stringMjs = require("../../gleam_stdlib/gleam/string.mjs");
var _gleamMjs = require("../gleam.mjs");
var _attributeMjs = require("../lustre/attribute.mjs");
var _effectMjs = require("../lustre/effect.mjs");
var _elementMjs = require("../lustre/element.mjs");
var _htmlMjs = require("../lustre/element/html.mjs");
var _appMjs = require("../lustre/runtime/app.mjs");
var _vattrMjs = require("../lustre/vdom/vattr.mjs");
var _componentFfiMjs = require("./runtime/client/component.ffi.mjs");
function on_attribute_change(name, decoder) {
    return new (0, _appMjs.Option)((config)=>{
        let attributes = (0, _gleamMjs.prepend)([
            name,
            decoder
        ], config.attributes);
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_property_change(name, decoder) {
    return new (0, _appMjs.Option)((config)=>{
        let properties = (0, _gleamMjs.prepend)([
            name,
            decoder
        ], config.properties);
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_context_change(key, decoder) {
    return new (0, _appMjs.Option)((config)=>{
        let contexts = (0, _gleamMjs.prepend)([
            key,
            decoder
        ], config.contexts);
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function form_associated() {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, true, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_form_autofill(handler) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, true, new (0, _optionMjs.Some)(handler), config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_form_reset(message) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, true, config.on_form_autofill, new (0, _optionMjs.Some)(message), config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_form_restore(handler) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, true, config.on_form_autofill, config.on_form_reset, new (0, _optionMjs.Some)(handler), config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function open_shadow_root(open) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(open, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function adopt_styles(adopt) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, adopt, config.delegates_focus, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function delegates_focus(delegates) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, delegates, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, config.on_disconnect);
    });
}
function on_connect(message) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, new (0, _optionMjs.Some)(message), config.on_adopt, config.on_disconnect);
    });
}
function on_adopt(message) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, new (0, _optionMjs.Some)(message), config.on_disconnect);
    });
}
function on_disconnect(message) {
    return new (0, _appMjs.Option)((config)=>{
        return new (0, _appMjs.Config)(config.open_shadow_root, config.adopt_styles, config.delegates_focus, config.attributes, config.properties, config.contexts, config.is_form_associated, config.on_form_autofill, config.on_form_reset, config.on_form_restore, config.on_connect, config.on_adopt, new (0, _optionMjs.Some)(message));
    });
}
function default_slot(attributes, fallback) {
    return _htmlMjs.slot(attributes, fallback);
}
function named_slot(name, attributes, fallback) {
    return _htmlMjs.slot((0, _gleamMjs.prepend)((0, _attributeMjs.attribute)("name", name), attributes), fallback);
}
function part(name) {
    return (0, _attributeMjs.attribute)("part", name);
}
function do_parts(loop$names, loop$part) {
    while(true){
        let names = loop$names;
        let part = loop$part;
        if (names instanceof (0, _gleamMjs.Empty)) return part;
        else {
            let $ = names.head[1];
            if ($) {
                let rest = names.tail;
                let name = names.head[0];
                return part + name + " " + do_parts(rest, part);
            } else {
                let rest = names.tail;
                loop$names = rest;
                loop$part = part;
            }
        }
    }
}
function parts(names) {
    return part(do_parts(names, ""));
}
function exportparts(names) {
    return (0, _attributeMjs.attribute)("exportparts", _stringMjs.join(names, ", "));
}
function slot(name) {
    return (0, _attributeMjs.attribute)("slot", name);
}
function set_form_value(value) {
    return _effectMjs.before_paint((_, root)=>{
        return (0, _componentFfiMjs.set_form_value)(root, value);
    });
}
function clear_form_value() {
    return _effectMjs.before_paint((_, root)=>{
        return (0, _componentFfiMjs.clear_form_value)(root);
    });
}
function set_pseudo_state(value) {
    return _effectMjs.before_paint((_, root)=>{
        return (0, _componentFfiMjs.set_pseudo_state)(root, value);
    });
}
function remove_pseudo_state(value) {
    return _effectMjs.before_paint((_, root)=>{
        return (0, _componentFfiMjs.remove_pseudo_state)(root, value);
    });
}
function prerender(component, tag, attributes, children) {
    let $ = _listMjs.fold(attributes, component.init(undefined), (state, attribute)=>{
        if (attribute instanceof (0, _vattrMjs.Attribute)) {
            let name = attribute.name;
            let value = attribute.value;
            let $1 = _listMjs.key_find(component.config.attributes, name);
            if ($1 instanceof (0, _gleamMjs.Ok)) {
                let handler = $1[0];
                let $2 = handler(value);
                if ($2 instanceof (0, _gleamMjs.Ok)) {
                    let message = $2[0];
                    return component.update(state[0], message);
                } else return state;
            } else return state;
        } else if (attribute instanceof (0, _vattrMjs.Property)) return state;
        else return state;
    });
    let model;
    model = $[0];
    let shadowrootmode = _attributeMjs.shadowrootmode((()=>{
        let $1 = component.config.open_shadow_root;
        if ($1) return "open";
        else return "closed";
    })());
    let shadowrootdelegatesfocus = _attributeMjs.shadowrootdelegatesfocus(component.config.delegates_focus);
    return _elementMjs.element(tag, attributes, (0, _gleamMjs.prepend)(_htmlMjs.template((0, _gleamMjs.toList)([
        shadowrootmode,
        shadowrootdelegatesfocus
    ]), (0, _gleamMjs.toList)([
        component.view(model)
    ])), children));
}

},{"../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../gleam_stdlib/gleam/string.mjs":"aB8qb","../gleam.mjs":"jNPQG","../lustre/attribute.mjs":"faRXj","../lustre/effect.mjs":"iAEPi","../lustre/element.mjs":"2XxJ4","../lustre/element/html.mjs":"eLT3l","../lustre/runtime/app.mjs":"fnyl8","../lustre/vdom/vattr.mjs":"jrrcC","./runtime/client/component.ffi.mjs":"eGPg4","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"faRXj":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Create an HTML attribute. This is like saying `element.setAttribute("class", "wibble")`
 * in JavaScript. Attributes will be rendered when calling [`element.to_string`](./element.html#to_string).
 *
 * > **Note**: there is a subtle difference between attributes and properties. You
 * > can read more about the implications of this
 * > [here](https://github.com/lustre-labs/lustre/blob/main/pages/hints/attributes-vs-properties.md).
 */ parcelHelpers.export(exports, "attribute", ()=>attribute);
/**
 * Create a DOM property. This is like saying `element.className = "wibble"` in
 * JavaScript. Properties will be **not** be rendered when calling
 * [`element.to_string`](./element.html#to_string).
 *
 * > **Note**: there is a subtle difference between attributes and properties. You
 * > can read more about the implications of this
 * > [here](https://github.com/lustre-labs/lustre/blob/main/pages/hints/attributes-vs-properties.md).
 */ parcelHelpers.export(exports, "property", ()=>property);
/**
 * Defines a shortcut key to activate or focus the element. Multiple options
 * may be provided as a set of space-separated characters that are exactly one
 * code point each.
 *
 * The way to activate the access key depends on the browser and its platform:
 *
 * |         | Windows           | Linux               | Mac OS              |
 * |---------|-------------------|---------------------|---------------------|
 * | Firefox | Alt + Shift + key | Alt + Shift + key   | Ctrl + Option + key |
 * | Chrome  | Alt + key         | Ctrl + Option + key | Ctrl + Option + key |
 * | Safari  |                   |                     | Ctrl + Option + key |
 */ parcelHelpers.export(exports, "accesskey", ()=>accesskey);
/**
 * Controls whether text input is automatically capitalised. The following values
 * are accepted:
 *
 * | Value        | Mode       |
 * |--------------|------------|
 * | ""           | default    |
 * | "none"       | none       |
 * | "off"        |            |
 * | "sentences"  | sentences  |
 * | "on"         |            |
 * | "words"      | words      |
 * | "characters" | characters |
 *
 * The autocapitalisation processing model is based on the following five modes:
 *
 * - **default**: The user agent and input method should make their own determination
 *   of whether or not to enable autocapitalization.
 *
 * - **none**: No autocapitalisation should be applied (all letters should default
 *   to lowercase).
 *
 * - **sentences**: The first letter of each sentence should default to a capital
 *   letter; all other letters should default to lowercase.
 *
 * - **words**: The first letter of each word should default to a capital letter;
 *   all other letters should default to lowercase.
 *
 * - **characters**: All letters should default to uppercase.
 */ parcelHelpers.export(exports, "autocapitalize", ()=>autocapitalize);
/**
 * Controls whether the user agent may automatically correct mispelled words
 * while typing. Whether or not spelling is corrected is left up to the user
 * agent and may also depend on the user's settings.
 *
 * When disabled the user agent is **never** allowed to correct spelling.
 */ parcelHelpers.export(exports, "autocorrect", ()=>autocorrect);
/**
 * For server-rendered HTML, this attribute controls whether an element should
 * be focused when the page first loads.
 *
 * > **Note**: Lustre's runtime augments that native behaviour of this attribute.
 * > Whenever it is toggled true, the element will be automatically focused even
 * > if it already exists in the DOM.
 */ parcelHelpers.export(exports, "autofocus", ()=>autofocus);
/**
 * A class is a non-unique identifier for an element primarily used for styling
 * purposes. You can provide multiple classes as a space-separated list and any
 * style rules that apply to any of the classes will be applied to the element.
 *
 * To conditionally toggle classes on and off, you can use the [`classes`](#classes)
 * function instead.
 *
 * > **Note**: unlike most attributes, multiple `class` attributes are merged
 * > with any existing other classes on an element. Classes added _later_ in the
 * > list will override classes added earlier.
 */ parcelHelpers.export(exports, "class$", ()=>class$);
/**
 * Create an empty attribute. This is not added to the DOM and not rendered when
 * calling [`element.to_string`](./element.html#to_string), but it is useful for
 * _conditionally_ adding attributes to an element.
 */ parcelHelpers.export(exports, "none", ()=>none);
/**
 * A class is a non-unique identifier for an element primarily used for styling
 * purposes. You can provide multiple classes as a space-separated list and any
 * style rules that apply to any of the classes will be applied to the element.
 * This function allows you to conditionally toggle classes on and off.
 *
 * > **Note**: unlike most attributes, multiple `class` attributes are merged
 * > with any existing other classes on an element. Classes added _later_ in the
 * > list will override classes added earlier.
 */ parcelHelpers.export(exports, "classes", ()=>classes);
/**
 * Indicates whether the element's content is editable by the user, allowing them
 * to modify the HTML content directly. The following values are accepted:
 *
 * | Value        | Description                                           |
 * |--------------|-------------------------------------------------------|
 * | "true"       | The element is editable.                              |
 * | ""           |                                                       |
 * | "false"      | The element is not editable.                          |
 * | "plain-text" | The element is editable without rich text formatting. |
 *
 * > **Note**: setting the value to an empty string does *not* disable this
 * > attribute, and is instead equivalent to setting it to `"true"`!
 */ parcelHelpers.export(exports, "contenteditable", ()=>contenteditable);
/**
 * Add a `data-*` attribute to an HTML element. The key will be prefixed by
 * `"data-"`, and accessible from JavaScript or in Gleam decoders under the
 * path `element.dataset.key` where `key` is the key you provide to this
 * function.
 */ parcelHelpers.export(exports, "data", ()=>data);
/**
 * Specifies the text direction of the element's content. The following values
 * are accepted:
 *
 * | Value  | Description                                                          |
 * |--------|----------------------------------------------------------------------|
 * | "ltr"  | The element's content is left-to-right.                              |
 * | "rtl"  | The element's content is right-to-left.                              |
 * | "auto" | The element's content direction is determined by the content itself. |
 *
 * > **Note**: the `"auto"` value should only be used as a last resort in cases
 * > where the content's direction is truly unknown. The heuristic used by
 * > browsers is naive and only considers the first character available that
 * > indicates the direction.
 */ parcelHelpers.export(exports, "dir", ()=>dir);
/**
 * Indicates whether the element can be dragged as part of the HTML drag-and-drop
 * API.
 */ parcelHelpers.export(exports, "draggable", ()=>draggable);
/**
 * Specifies what action label (or potentially icon) to present for the "enter"
 * key on virtual keyboards such as mobile devices. The following values are
 * accepted:
 *
 * | Value      | Example        |
 * |------------|----------------|
 * | "enter"    | "return", "↵"  |
 * | "done"     | "done", "✅"   |
 * | "go"       | "go"           |
 * | "next"     | "next"         |
 * | "previous" | "return"       |
 * | "search"   | "search", "🔍" |
 * | "send"     | "send"         |
 *
 * The examples listed are demonstrative and may not be the actual labels used
 * by user agents. When unspecified or invalid, the user agent may use contextual
 * information such as the type of an input to determine the label.
 */ parcelHelpers.export(exports, "enterkeyhint", ()=>enterkeyhint);
/**
 * Indicates whether the element is relevant to the page's current state. A
 * hidden element is not visible to the user and is inaccessible to assistive
 * technologies such as screen readers. This makes it unsuitable for simple
 * presentation purposes, but it can be useful for example to render something
 * that may be made visible later.
 */ parcelHelpers.export(exports, "hidden", ()=>hidden);
/**
 * The `"id"` attribute is used to uniquely identify a single element within a
 * document. It can be used to reference the element in CSS with the selector
 * `#id`, in JavaScript with `document.getElementById("id")`, or by anchors on
 * the same page with the URL `"#id"`.
 */ parcelHelpers.export(exports, "id", ()=>id);
/**
 * Marks the element as inert, meaning it is not currently interactive and does
 * not receive user input. For sighted users, it's common to style inert elements
 * in a way that makes them visually distinct from active elements, such as by
 * greying them out: this can help avoid confusion for users who may not otherwise
 * know the content they are looking at is inactive.
 */ parcelHelpers.export(exports, "inert", ()=>inert);
/**
 * Hints to the user agent about what type of virtual keyboard to display when
 * the user interacts with the element. The following values are accepted:
 *
 * | Value        | Description                                                   |
 * |--------------|---------------------------------------------------------------|
 * | "none"       | No virtual keyboard should be displayed.                      |
 * | "text"       | A standard text input keyboard.                               |
 * | "decimal"    | A numeric keyboard with locale-appropriate separator.         |
 * | "numeric"    | A numeric keyboard.                                           |
 * | "tel"        | A telephone keypad including "#" and "*".                     |
 * | "email"      | A keyboard for entering email addresses including "@" and "." |
 * | "url"        | A keyboard for entering URLs including "/" and ".".           |
 * | "search"     | A keyboard for entering search queries should be shown.       |
 *
 * The `"none"` value should only be used in cases where you are rendering a
 * custom input method, otherwise the user will not be able to enter any text!
 */ parcelHelpers.export(exports, "inputmode", ()=>inputmode);
/**
 * Specifies the [customised built-in element](https://html.spec.whatwg.org/#customized-built-in-element)
 * to be used in place of the native element this attribute is applied to.
 */ parcelHelpers.export(exports, "is", ()=>is);
/**
 * Used as part of the [Microdata](https://schema.org/docs/gs.html) format to
 * specify the global unique identifier of an item, for example books that are
 * identifiable by their ISBN.
 */ parcelHelpers.export(exports, "itemid", ()=>itemid);
/**
 * Used as part of the [Microdata](https://schema.org/docs/gs.html) format to
 * specify that the content of the element is to be treated as a value of the
 * given property name.
 */ parcelHelpers.export(exports, "itemprop", ()=>itemprop);
/**
 * Used as part of the [Microdata](https://schema.org/docs/gs.html) format to
 * indicate that the element and its descendants form a single item of key-value
 * data.
 */ parcelHelpers.export(exports, "itemscope", ()=>itemscope);
/**
 * Used as part of the [Microdata](https://schema.org/docs/gs.html) format to
 * specify the type of item being described. This is a URL that points to
 * a schema containing the vocabulary used for an item's key-value pairs, such
 * as a schema.org type.
 */ parcelHelpers.export(exports, "itemtype", ()=>itemtype);
/**
 * Specifies the language of the element's content and the language of any of
 * this element's attributes that contain text. The `"lang"` attribute applies
 * to the element itself and all of its descendants, unless overridden by
 * another `"lang"` attribute on a descendant element.
 *
 * The value must be a valid [BCP 47 language tag](https://tools.ietf.org/html/bcp47).
 */ parcelHelpers.export(exports, "lang", ()=>lang);
/**
 * A cryptographic nonce used by CSP (Content Security Policy) to allow or
 * deny the fetch of a given resource.
 */ parcelHelpers.export(exports, "nonce", ()=>nonce);
/**
 * Specifies that the element should be treated as a popover, rendering it in
 * the top-layer above all other content when the popover is active. The following
 * values are accepted:
 *
 * | Value        | Description                                    |
 * |--------------|------------------------------------------------|
 * | "auto"       | Closes other popovers when opened.             |
 * | ""           |                                                |
 * | "manual"     | Does not close other popovers when opened.     |
 * | "hint"       | Closes only other "hint" popovers when opened. |
 *
 * All modes except `"manual"` support "light dismiss" letting the user close
 * the popover by clicking outside of it, as well as respond to close requests
 * letting the user dismiss a popover by pressing the "escape" key or by using
 * the dismiss gesture on any assistive technology.
 *
 * Popovers can be triggered either programmatically through the `showPopover()`
 * method, or by assigning an [`id`](#id) to the element and including the
 * [`popovertarget`](#popovertarget) attribute on the element that should trigger
 * the popover.
 */ parcelHelpers.export(exports, "popover", ()=>popover);
/**
 * Indicates whether the element's content should be checked for spelling errors.
 * This typically only applies to inputs and textareas, or elements that are
 * [`contenteditable`](#contenteditable).
 */ parcelHelpers.export(exports, "spellcheck", ()=>spellcheck);
/**
 * Provide a single property name and value to be used as inline styles for the
 * element. If either the property name or value is empty, this attribute will
 * be ignored.
 *
 * > **Note**: unlike most attributes, multiple `style` attributes are merged
 * > with any existing other styles on an element. Styles added _later_ in the
 * > list will override styles added earlier.
 */ parcelHelpers.export(exports, "style", ()=>style);
/**
 * Provide a list of property-value pairs to be used as inline styles for the
 * element. Empty properties or values are omitted from the final style string.
 *
 * > **Note**: unlike most attributes, multiple `styles` attributes are merged
 * > with any existing other styles on an element. Styles added _later_ in the
 * > list will override styles added earlier.
 */ parcelHelpers.export(exports, "styles", ()=>styles);
/**
 * Specifies the tabbing order of the element. If an element is not typically
 * focusable, such as a `<div>`, it will be made focusable when this attribute
 * is set.
 *
 * Any integer value is accepted, but the following values are recommended:
 *
 * - `-1`: indicates the element may receive focus, but should not be sequentially
 *   focusable. The user agent may choose to ignore this preference if, for
 *   example, the user agent is a screen reader.
 *
 * - `0`: indicates the element may receive focus and should be placed in the
 *   sequential focus order in the order it appears in the DOM.
 *
 * - any positive integer: indicates the element should be placed in the sequential
 *   focus order relative to other elements with a positive tabindex.
 *
 * Values other than `0` and `-1` are generally not recommended as managing
 * the relative order of focusable elements can be difficult and error-prone.
 */ parcelHelpers.export(exports, "tabindex", ()=>tabindex);
/**
 * Annotate an element with additional information that may be suitable as a
 * tooltip, such as a description of a link or image.
 *
 * It is **not** recommended to use the `title` attribute as a way of providing
 * accessibility information to assistive technologies. User agents often do not
 * expose the `title` attribute to keyboard-only users or touch devices, for
 * example.
 */ parcelHelpers.export(exports, "title", ()=>title);
/**
 * Controls whether an element's content may be translated by the user agent
 * when the page is localised. This includes both the element's text content
 * and some of its attributes:
 *
 * | Attribute   | Element(s)                                 |
 * |-------------|--------------------------------------------|
 * | abbr        | th                                         |
 * | alt         | area, img, input                           |
 * | content     | meta                                       |
 * | download    | a, area                                    |
 * | label       | optgroup, option, track                    |
 * | lang        | *                                          |
 * | placeholder | input, textarea                            |
 * | srcdoc      | iframe                                     |
 * | title       | *                                          |
 * | style       | *                                          |
 * | value       | input (with type="button" or type="reset") |
 */ parcelHelpers.export(exports, "translate", ()=>translate);
/**
 * Indicates if writing suggestions should be enabled for this element.
 */ parcelHelpers.export(exports, "writingsuggestions", ()=>writingsuggestions);
/**
 * Indicates whether the details element is open or closed.
 */ parcelHelpers.export(exports, "open", ()=>open);
/**
 * Specifies the URL of a linked resource. This attribute can be used on various
 * elements to create hyperlinks or to load resources.
 */ parcelHelpers.export(exports, "href", ()=>href);
/**
 * Specifies where to display the linked resource or where to open the link.
 * The following values are accepted:
 *
 * | Value     | Description                                             |
 * |-----------|---------------------------------------------------------|
 * | "_self"   | Open in the same frame/window (default)                 |
 * | "_blank"  | Open in a new window or tab                             |
 * | "_parent" | Open in the parent frame                                |
 * | "_top"    | Open in the full body of the window                     |
 * | framename | Open in a named frame                                   |
 *
 * > **Note**: consider against using `"_blank"` for links to external sites as it
 * > removes user control over their browsing experience.
 */ parcelHelpers.export(exports, "target", ()=>target);
/**
 * Indicates that the linked resource should be downloaded rather than displayed.
 * When provided with a value, it suggests a filename for the downloaded file.
 */ parcelHelpers.export(exports, "download", ()=>download);
/**
 * Provides a space-separated list of URLs that will be notified if the user
 * follows the hyperlink. These URLs will receive POST requests with bodies
 * of type `ping/1.0`.
 */ parcelHelpers.export(exports, "ping", ()=>ping);
/**
 * Specifies the relationship between the current document and the linked resource.
 * Multiple relationship values can be provided as a space-separated list.
 */ parcelHelpers.export(exports, "rel", ()=>rel);
/**
 * Specifies the language of the linked resource. The value must be a valid
 * [BCP 47 language tag](https://tools.ietf.org/html/bcp47).
 */ parcelHelpers.export(exports, "hreflang", ()=>hreflang);
/**
 * Specifies the referrer policy for fetches initiated by the element. The
 * following values are accepted:
 *
 * | Value                              | Description                                           |
 * |-----------------------------------|--------------------------------------------------------|
 * | "no-referrer"                     | No Referer header is sent                              |
 * | "no-referrer-when-downgrade"      | Only send Referer for same-origin or more secure       |
 * | "origin"                          | Send only the origin part of the URL                   |
 * | "origin-when-cross-origin"        | Full URL for same-origin, origin only for cross-origin |
 * | "same-origin"                     | Only send Referer for same-origin requests             |
 * | "strict-origin"                   | Like origin, but only to equally secure destinations   |
 * | "strict-origin-when-cross-origin" | Default policy with varying levels of restriction      |
 * | "unsafe-url"                      | Always send the full URL                               |
 */ parcelHelpers.export(exports, "referrerpolicy", ()=>referrerpolicy);
/**
 * Specifies the type of the resource being linked to, which is necessary for
 * request matching, application of correct content security policy, and setting
 * of correct Accept request header.
 *
 * > **Note**: this attribute is required when rel="preload" has been set on the
 * > `<link>` element, optional when `rel="modulepreload"` has been set, and
 * > otherwise should not be used.
 *
 * | Value      | Applies to                       |
 * |------------|----------------------------------|
 * | "audio"    | `<audio>`                        |
 * | "document" | `<iframe>`, `<frame>`            |
 * | "embed"    | `<embed>`                        |
 * | "fetch"    | fetch, XHR                       |
 * | "font"     | CSS @font-face                   |
 * | "image"    | `<img>`, `<image>`, `<picture>`  |
 * | "object"   | `<object>`                       |
 * | "script"   | `<script>`, Worker importScripts |
 * | "style"    | `<link rel="stylesheet">`        |
 * | "video"    | `<video>`                        |
 * | "worker"   | Worker, SharedWorker             |
 */ parcelHelpers.export(exports, "as_", ()=>as_);
/**
 * This attribute explicitly indicates that certain operations should be blocked
 * until specific conditions are met. It must only be used when the rel attribute
 * contains the expect or stylesheet keywords. With `rel="expect"`, it indicates
 * that operations should be blocked until a specific DOM node has been parsed.
 * With `rel="stylesheet"`, it indicates that operations should be blocked until
 * an external stylesheet and its critical subresources have been fetched and
 * applied to the document.
 */ parcelHelpers.export(exports, "blocking", ()=>blocking);
/**
 * Provides a base64-encoded hash of the resource being linked to. This is used
 * by the browser to verify that a fetched resource has not been tampered with.
 *
 * > **Note**: this attribute is only meaningful on `<link>` elements with either
 * > `rel="stylesheet"`, `rel="preload"`, or `rel="modulepreload"`. It may also
 * > be used on `<script>` elements.
 */ parcelHelpers.export(exports, "integrity", ()=>integrity);
/**
 * Specifies text that should be displayed when the image cannot be rendered.
 * This attribute is essential for accessibility, providing context about the
 * image to users who cannot see it, including those using screen readers.
 */ parcelHelpers.export(exports, "alt", ()=>alt);
/**
 * Specifies the URL of an image or resource to be used.
 */ parcelHelpers.export(exports, "src", ()=>src);
/**
 * Specifies a set of image sources for different display scenarios. This allows
 * browsers to choose the most appropriate image based on factors like screen
 * resolution and viewport size.
 */ parcelHelpers.export(exports, "srcset", ()=>srcset);
/**
 * Used with `srcset` to define the size of images in different layout scenarios.
 * Helps the browser select the most appropriate image source.
 */ parcelHelpers.export(exports, "sizes", ()=>sizes);
/**
 * Configures the CORS (Cross-Origin Resource Sharing) settings for the element.
 * Valid values are "anonymous" and "use-credentials".
 */ parcelHelpers.export(exports, "crossorigin", ()=>crossorigin);
/**
 * Specifies the name of an image map to be used with the image.
 */ parcelHelpers.export(exports, "usemap", ()=>usemap);
/**
 * Indicates that the image is a server-side image map. When a user clicks on the
 * image, the coordinates of the click are sent to the server.
 */ parcelHelpers.export(exports, "ismap", ()=>ismap);
/**
 * Specifies the width of the element in pixels.
 */ parcelHelpers.export(exports, "width", ()=>width);
/**
 * Specifies the height of the element in pixels.
 */ parcelHelpers.export(exports, "height", ()=>height);
/**
 * Provides a hint about how the image should be decoded. Valid values are
 * "sync", "async", and "auto".
 */ parcelHelpers.export(exports, "decoding", ()=>decoding);
/**
 * Indicates how the browser should load the image. Valid values are "eager"
 * (load immediately) and "lazy" (defer loading until needed).
 */ parcelHelpers.export(exports, "loading", ()=>loading);
/**
 * Sets the priority for fetches initiated by the element. Valid values are
 * "high", "low", and "auto".
 */ parcelHelpers.export(exports, "fetchpriority", ()=>fetchpriority);
/**
 * Specifies the character encodings to be used for form submission. This allows
 * servers to know how to interpret the form data. Multiple encodings can be
 * specified as a space-separated list.
 */ parcelHelpers.export(exports, "accept_charset", ()=>accept_charset);
/**
 * Specifies the URL to which the form's data should be sent when submitted.
 * This can be overridden by formaction attributes on submit buttons.
 */ parcelHelpers.export(exports, "action", ()=>action);
/**
 * Specifies how form data should be encoded before sending it to the server.
 * Valid values include:
 *
 * | Value                               | Description                           |
 * |-------------------------------------|---------------------------------------|
 * | "application/x-www-form-urlencoded" | Default encoding (spaces as +, etc.)  |
 * | "multipart/form-data"               | Required for file uploads             |
 * | "text/plain"                        | Simple encoding with minimal escaping  |
 */ parcelHelpers.export(exports, "enctype", ()=>enctype);
/**
 * Specifies the HTTP method to use when submitting the form. Common values are:
 *
 * | Value    | Description                                              |
 * |----------|----------------------------------------------------------|
 * | "get"    | Appends form data to URL (default)                       |
 * | "post"   | Sends form data in the body of the HTTP request          |
 * | "dialog" | Closes a dialog if the form is inside one                |
 */ parcelHelpers.export(exports, "method", ()=>method);
/**
 * When present, indicates that the form should not be validated when submitted.
 * This allows submission of forms with invalid or incomplete data.
 */ parcelHelpers.export(exports, "novalidate", ()=>novalidate);
/**
 * A hint for the user agent about what file types are expected to be submitted.
 * The following values are accepted:
 *
 * | Value     | Description                                          |
 * |-----------|------------------------------------------------------|
 * | "audio/*" | Any audio file type.                                 |
 * | "video/*" | Any video file type.                                 |
 * | "image/*" | Any image file type.                                 |
 * | mime/type | A complete MIME type, without additional parameters. |
 * | .ext      | Indicates any file with the given extension.         |
 *
 * The following input types support the `"accept"` attribute:
 *
 * - `"file"`
 *
 * > **Note**: the `"accept"` attribute is a *hint* to the user agent and does
 * > not guarantee that the user will only be able to select files of the specified
 * > type.
 */ parcelHelpers.export(exports, "accept", ()=>accept);
/**
 * Allow a colour's alpha component to be manipulated, allowing the user to
 * select a colour with transparency.
 *
 * The following input types support the `"alpha"` attribute:
 *
 * - `"color"`
 */ parcelHelpers.export(exports, "alpha", ()=>alpha);
/**
 * A hint for the user agent to automatically fill the value of the input with
 * an appropriate value. The format for the `"autocomplete"` attribute is a
 * space-separated ordered list of optional tokens:
 *
 *     "section-* (shipping | billing) [...fields] webauthn"
 *
 * - `section-*`: used to disambiguate between two fields with otherwise identical
 *   autocomplete values. The `*` is replaced with a string that identifies the
 *   section of the form.
 *
 * - `shipping | billing`: indicates the field is related to shipping or billing
 *   address or contact information.
 *
 * - `[...fields]`: a space-separated list of field names that are relevant to
 *   the input, for example `"email"`, `"name family-name"`, or `"home tel"`.
 *
 * - `webauthn`: indicates the field can be automatically filled with a WebAuthn
 *   passkey.
 *
 * In addition, the value may instead be `"off"` to disable autocomplete for the
 * input, or `"on"` to let the user agent decide based on context what values
 * are appropriate.
 *
 * The following input types support the `"autocomplete"` attribute:
 *
 * - `"color"`
 * - `"date"`
 * - `"datetime-local"`
 * - `"email"`
 * - `"hidden"`
 * - `"month"`
 * - `"number"`
 * - `"password"`
 * - `"range"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"time"`
 * - `"url"`
 * - `"week"`
 */ parcelHelpers.export(exports, "autocomplete", ()=>autocomplete);
/**
 * Whether the control is checked or not. When participating in a form, the
 * value of the input is included in the form submission if it is checked. For
 * checkboxes that do not have a value, the value of the input is `"on"` when
 * checked.
 *
 * The following input types support the `"checked"` attribute:
 *
 * - `"checkbox"`
 * - `"radio"`
 */ parcelHelpers.export(exports, "checked", ()=>checked);
/**
 * Set the default checked state of a form control. This element will appear
 * checked to users when the input is first rendered and its value will included in the form
 * submission if the user does not change it.
 *
 * Just setting a default value and letting the DOM manage the state of an input
 * is known as using [_uncontrolled inputs_](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 * Doing this means your application cannot set the value of an input after it
 * is modified without using an effect.
 */ parcelHelpers.export(exports, "default_checked", ()=>default_checked);
/**
 * The color space of the serialised CSS color. It also hints to user agents
 * about what kind of interface to present to the user for selecting a color.
 * The following values are accepted:
 *
 * - `"limited-srgb"`: The CSS color is converted to the 'srgb' color space and
 *   limited to 8-bits per component, e.g., `"#123456"` or
 *   `"color(srgb 0 1 0 / 0.5)"`.
 *
 * - `"display-p3"`: The CSS color is converted to the 'display-p3' color space,
 *   e.g., `"color(display-p3 1.84 -0.19 0.72 / 0.6)"`.
 *
 * The following input types support the `"colorspace"` attribute:
 *
 * - `"color"`
 */ parcelHelpers.export(exports, "colorspace", ()=>colorspace);
/**
 * A positive integer value indicating how many visible columns the text control
 * will have. The default value is 20.
 */ parcelHelpers.export(exports, "cols", ()=>cols);
/**
 * The name of the field included in a form that indicates the direcionality of
 * the user's input.
 *
 * The following input types support the `"dirname"` attribute:
 *
 * - `"email"`
 * - `"hidden"`
 * - `"password"`
 * - `"search"`
 * - `"submit"
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "dirname", ()=>dirname);
/**
 * Controls whether or not the input is disabled. Disabled inputs are not
 * validated during form submission and are not interactive.
 */ parcelHelpers.export(exports, "disabled", ()=>disabled);
/**
 *
 */ parcelHelpers.export(exports, "for$", ()=>for$);
/**
 * Associates the input with a form element located elsewhere in the document.
 */ parcelHelpers.export(exports, "form", ()=>form);
/**
 * The URL to use for form submission. This URL will override the [`"action"`](#action)
 * attribute on the form element itself, if present.
 *
 * The following input types support the `"formaction"` attribute:
 *
 * - `"image"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "formaction", ()=>formaction);
/**
 * Entry list encoding type to use for form submission
 *
 * - `"image"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "formenctype", ()=>formenctype);
/**
 * Variant to use for form submission
 *
 * - `"image"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "formmethod", ()=>formmethod);
/**
 * Bypass form control validation for form submission
 *
 * - `"image"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "formnovalidate", ()=>formnovalidate);
/**
 * Navigable for form submission
 *
 * - `"image"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "formtarget", ()=>formtarget);
/**
 * List of autocomplete options
 *
 * The following input types support the `"list"` attribute:
 *
 * - `"color"`
 * - `"date"`
 * - `"datetime-local"`
 * - `"email"`
 * - `"month"`
 * - `"number"`
 * - `"range"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"time"`
 * - `"url"`
 * - `"week"`
 */ parcelHelpers.export(exports, "list", ()=>list);
/**
 * Constrain the maximum value of a form control. The exact syntax of this value
 * changes depending on the type of input, for example `"1"`, `"1979-12-31"`, and
 * `"06:00"` are all potentially valid values for the `"max"` attribute.
 *
 * The following input types support the `"max"` attribute:
 *
 * - `"date"`
 * - `"datetime-local"`
 * - `"month"`
 * - `"number"`
 * - `"range"`
 * - `"time"`
 * - `"week"`
 */ parcelHelpers.export(exports, "max", ()=>max);
/**
 * Maximum length of value
 *
 * The following input types support the `"maxlength"` attribute:
 *
 * - `"email"`
 * - `"password"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "maxlength", ()=>maxlength);
/**
 * Minimum value
 *
 * The following input types support the `"max"` attribute:
 *
 * - `"date"`
 * - `"datetime-local"`
 * - `"month"`
 * - `"number"`
 * - `"range"`
 * - `"time"`
 * - `"week"`
 */ parcelHelpers.export(exports, "min", ()=>min);
/**
 * Minimum length of value
 *
 * - `"email"`
 * - `"password"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "minlength", ()=>minlength);
/**
 * Whether an input or select may allow multiple values to be selected at once.
 *
 * The following input types support the `"multiple"` attribute:
 *
 * - `"email"`
 * - `"file"`
 */ parcelHelpers.export(exports, "multiple", ()=>multiple);
/**
 * Name of the element to use for form submission and in the form.elements API
 */ parcelHelpers.export(exports, "name", ()=>name);
/**
 * Pattern to be matched by the form control's value
 *
 * - `"email"`
 * - `"password"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "pattern", ()=>pattern);
/**
 * User-visible label to be placed within the form control
 *
 * - `"email"`
 * - `"number"`
 * - `"password"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "placeholder", ()=>placeholder);
/**
 * Targets a popover element to toggle, show, or hide
 *
 * The following input types support the `"popovertarget"` attribute:
 *
 * - `"button"`
 * - `"image"`
 * - `"reset"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "popovertarget", ()=>popovertarget);
/**
 * Indicates whether a targeted popover element is to be toggled, shown, or hidden
 *
 * The following input types support the `"popovertarget"` attribute:
 *
 * - `"button"`
 * - `"image"`
 * - `"reset"`
 * - `"submit"`
 */ parcelHelpers.export(exports, "popovertargetaction", ()=>popovertargetaction);
/**
 * Whether to allow the value to be edited by the user
 *
 * - `"date"`
 * - `"datetime-local"`
 * - `"email"`
 * - `"month"`
 * - `"number"`
 * - `"password"`
 * - `"range"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"time"`
 * - `"url"`
 * - `"week"`
 */ parcelHelpers.export(exports, "readonly", ()=>readonly);
/**
 * Whether the control is required for form submission
 *
 * - `"checkbox"`
 * - `"date"`
 * - `"datetime-local"`
 * - `"email"`
 * - `"month"`
 * - `"number"`
 * - `"password"`
 * - `"radio"`
 * - `"range"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"time"`
 * - `"url"`
 * - `"week"`
 */ parcelHelpers.export(exports, "required", ()=>required);
/**
 * A positive integer value indicating how many visible rows the text control
 * will have. The browsers default value is 2.
 */ parcelHelpers.export(exports, "rows", ()=>rows);
/**
 * Controls whether or not a select's `<option>` is selected or not. Only one
 * option can be selected at a time, unless the [`"multiple"`](#multiple)
 * attribute is set on the select element.
 */ parcelHelpers.export(exports, "selected", ()=>selected);
/**
 * An `<option>` with this attribute toggled on will be selected when
 * its corresponding select is rendered for the first time. Only one
 * option can be selected at a time, unless the [`"multiple"`](#multiple)
 * attribute is set on the select element.
 *
 * Just setting a default value and letting the DOM manage the state of an input
 * is known as using [_uncontrolled inputs_](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 * Doing this means your application cannot set the value of an input after it
 * is modified without using an effect.
 */ parcelHelpers.export(exports, "default_selected", ()=>default_selected);
/**
 * Size of the control
 *
 * The following input types support the `size` attribute:
 *
 * - `"email"`
 * - `"password"`
 * - `"search"`
 * - `"tel"`
 * - `"text"`
 * - `"url"`
 */ parcelHelpers.export(exports, "size", ()=>size);
/**
 * Granularity to be matched by the form control's value
 *
 * The following input types support the `"step"` attribute:
 *
 * - `"date"`
 * - `"datetime-local"`
 * - `"month"`
 * - `"number"`
 * - `"range"`
 * - `"time"`
 * - `"week"`
 */ parcelHelpers.export(exports, "step", ()=>step);
/**
 * Type of form control
 */ parcelHelpers.export(exports, "type_", ()=>type_);
/**
 * Specifies the value of an input or form control. Using this attribute will
 * make sure the value is always in sync with your application's modelled, a
 * practice known as [_controlled inputs_](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 *
 * If you'd like to let the DOM manage the value of an input but still set a
 * default value for users to see, use the [`default_value`](#default_value)
 * attribute instead.
 */ parcelHelpers.export(exports, "value", ()=>value);
/**
 * Set the default value of an input or form control. This is the value that will
 * be shown to users when the input is first rendered and included in the form
 * submission if the user does not change it.
 *
 * Just setting a default value and letting the DOM manage the state of an input
 * is known as using [_uncontrolled inputs_](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 * Doing this means your application cannot set the value of an input after it
 * is modified without using an effect.
 */ parcelHelpers.export(exports, "default_value", ()=>default_value);
/**
 * Sets a pragma directive for a document. This is used in meta tags to define
 * behaviors the user agent should follow.
 */ parcelHelpers.export(exports, "http_equiv", ()=>http_equiv);
/**
 * Specifies the value of the meta element, which varies depending on the value
 * of the name or http-equiv attribute.
 */ parcelHelpers.export(exports, "content", ()=>content);
/**
 * Declares the character encoding used in the document. When used with a meta
 * element, this replaces the need for the `http_equiv("content-type")` attribute.
 */ parcelHelpers.export(exports, "charset", ()=>charset);
/**
 * Specifies the media types the resource applies to. This is commonly used with
 * link elements for stylesheets to determine when they should be loaded.
 */ parcelHelpers.export(exports, "media", ()=>media);
/**
 * Indicates that the media resource should automatically begin playing as soon
 * as it can do so without stopping. When not present, the media will not
 * automatically play until the user initiates playback.
 *
 * > **Note**: Lustre's runtime augments this attribute. Whenever it is toggled
 * > to true, the media will begin playing as if the element's `play()` method
 * > was called.
 */ parcelHelpers.export(exports, "autoplay", ()=>autoplay);
/**
 * When present, this attribute shows the browser's built-in control panel for the
 * media player, giving users control over playback, volume, seeking, and more.
 */ parcelHelpers.export(exports, "controls", ()=>controls);
/**
 * When present, this attribute indicates that the media should start over again
 * from the beginning when it reaches the end.
 */ parcelHelpers.export(exports, "loop", ()=>loop);
/**
 * When present, this attribute indicates that the audio output of the media element
 * should be initially silenced.
 */ parcelHelpers.export(exports, "muted", ()=>muted);
/**
 * Encourages the user agent to display video content within the element's
 * playback area rather than in a separate window or fullscreen, especially on
 * mobile devices.
 *
 * This attribute only acts as a *hint* to the user agent, and setting this to
 * false does not imply that the video will be played in fullscreen.
 */ parcelHelpers.export(exports, "playsinline", ()=>playsinline);
/**
 * Specifies an image to be shown while the video is downloading, or until the
 * user hits the play button.
 */ parcelHelpers.export(exports, "poster", ()=>poster);
/**
 * Provides a hint to the browser about what the author thinks will lead to the
 * best user experience. The following values are accepted:
 *
 * | Value      | Description                                                      |
 * |------------|------------------------------------------------------------------|
 * | "auto"     | Let's the user agent determine the best option                   |
 * | "metadata" | Hints to the user agent that it can fetch the metadata only.     |
 * | "none"     | Hints to the user agent that server traffic should be minimised. |
 */ parcelHelpers.export(exports, "preload", ()=>preload);
/**
 * Specifies the mode for creating a shadow root on a template. Valid values
 * include:
 *
 * | Value     | Description                                 |
 * |-----------|---------------------------------------------|
 * | "open"    | Shadow root's contents are accessible       |
 * | "closed"  | Shadow root's contents are not accessible   |
 *
 * > **Note**: if you are pre-rendering a Lustre component you must make sure this
 * > attribute matches the [`open_shadow_root`](./component.html#open_shadow_root)
 * > configuration - or `"closed"` if not explicitly set - to ensure the shadow
 * > root is created correctly.
 */ parcelHelpers.export(exports, "shadowrootmode", ()=>shadowrootmode);
/**
 * Indicates whether focus should be delegated to the shadow root when an element
 * in the shadow tree gains focus.
 */ parcelHelpers.export(exports, "shadowrootdelegatesfocus", ()=>shadowrootdelegatesfocus);
/**
 * Determines whether the shadow root can be cloned when the host element is
 * cloned.
 */ parcelHelpers.export(exports, "shadowrootclonable", ()=>shadowrootclonable);
/**
 * Controls whether the shadow root should be preserved during serialization
 * operations like copying to the clipboard or saving a page.
 */ parcelHelpers.export(exports, "shadowrootserializable", ()=>shadowrootserializable);
/**
 * A short, abbreviated description of the header cell's content provided as an
 * alternative label to use for the header cell when referencing the cell in other
 * contexts. Some user-agents, such as speech readers, may present this description
 * before the content itself.
 */ parcelHelpers.export(exports, "abbr", ()=>abbr);
/**
 * A non-negative integer value indicating how many columns the header cell spans
 * or extends. The default value is `1`. User agents dismiss values higher than
 * `1000` as incorrect, defaulting such values to `1`.
 */ parcelHelpers.export(exports, "colspan", ()=>colspan);
/**
 * A list of space-separated strings corresponding to the id attributes of the
 * `<th>` elements that provide the headers for this header cell.
 */ parcelHelpers.export(exports, "headers", ()=>headers);
/**
 * A non-negative integer value indicating how many rows the header cell spans
 * or extends. The default value is `1`; if its value is set to `0`, the header
 * cell will extends to the end of the table grouping section, that the `<th>`
 * belongs to. Values higher than `65534` are clipped at `65534`.
 */ parcelHelpers.export(exports, "rowspan", ()=>rowspan);
/**
 * Specifies the number of consecutive columns a `<colgroup>` element spans. The
 * value must be a positive integer greater than zero.
 */ parcelHelpers.export(exports, "span", ()=>span);
/**
 * The `scope` attribute specifies whether a header cell is a header for a row,
 * column, or group of rows or columns. The following values are accepted:
 *
 * The `scope` attribute is only valid on `<th>` elements.
 */ parcelHelpers.export(exports, "scope", ()=>scope);
/**
 * Indicates the time and/or date of a `<time>` element. Values may be one of
 * the following formats:
 *
 * | Description                       | Syntax                                                                                                                                     | Examples                                                                                                                                   |
 * |-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
 * | Valid month string                | `YYYY-MM`                                                                                                                                  | `2011-11`, `2013-05`                                                                                                                       |
 * | Valid date string                 | `YYYY-MM-DD`                                                                                                                               | `1887-12-01`                                                                                                                               |
 * | Valid local date and time string  | `YYYY-MM-DD HH:MM`, `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD HH:MM:SS.mmm`, `YYYY-MM-DDTHH:MM`, `YYYY-MM-DDTHH:MM:SS`, `YYYY-MM-DDTHH:MM:SS.mmm` | `2013-12-25 11:12`, `1972-07-25 13:43:07`, `1941-03-15 07:06:23.678`, `2013-12-25T11:12`, `1972-07-25T13:43:07`, `1941-03-15T07:06:23.678` |
 * | Valid global date and time string | A valid local date and time string followed by a valid time-zone offset string                                                             | `2013-12-25 11:12+0200`, `1972-07-25 13:43:07+04:30`, `1941-03-15 07:06:23.678Z`, `2013-12-25T11:12-08:00`                                 |
 * | Valid week string                 | `YYYY-WWW`                                                                                                                                 | `2013-W46`                                                                                                                                 |
 *
 * A comprehensive list of valid formats can be found on [MDN](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/time#valid_datetime_values).
 */ parcelHelpers.export(exports, "datetime", ()=>datetime);
/**
 * Add an `aria-*` attribute to an HTML element. The key will be prefixed by
 * `aria-`.
 */ parcelHelpers.export(exports, "aria", ()=>aria);
/**
 *
 */ parcelHelpers.export(exports, "role", ()=>role);
/**
 * The aria-activedescendant attribute identifies the currently active element
 * when focus is on a composite widget, combobox, textbox, group, or application.
 */ parcelHelpers.export(exports, "aria_activedescendant", ()=>aria_activedescendant);
/**
 * In ARIA live regions, the global aria-atomic attribute indicates whether
 * assistive technologies such as a screen reader will present all, or only parts
 * of, the changed region based on the change notifications defined by the
 * aria-relevant attribute.
 */ parcelHelpers.export(exports, "aria_atomic", ()=>aria_atomic);
/**
 * The aria-autocomplete attribute indicates whether inputting text could trigger
 * display of one or more predictions of the user's intended value for a combobox,
 * searchbox, or textbox and specifies how predictions will be presented if they
 * are made.
 */ parcelHelpers.export(exports, "aria_autocomplete", ()=>aria_autocomplete);
/**
 * The global aria-braillelabel property defines a string value that labels the
 * current element, which is intended to be converted into Braille.
 */ parcelHelpers.export(exports, "aria_braillelabel", ()=>aria_braillelabel);
/**
 * The global aria-brailleroledescription attribute defines a human-readable,
 * author-localized abbreviated description for the role of an element intended
 * to be converted into Braille.
 */ parcelHelpers.export(exports, "aria_brailleroledescription", ()=>aria_brailleroledescription);
/**
 * Used in ARIA live regions, the global aria-busy state indicates an element is
 * being modified and that assistive technologies may want to wait until the
 * changes are complete before informing the user about the update.
 */ parcelHelpers.export(exports, "aria_busy", ()=>aria_busy);
/**
 * The aria-checked attribute indicates the current "checked" state of checkboxes,
 * radio buttons, and other widgets.
 */ parcelHelpers.export(exports, "aria_checked", ()=>aria_checked);
/**
 * The aria-colcount attribute defines the total number of columns in a table,
 * grid, or treegrid when not all columns are present in the DOM.
 */ parcelHelpers.export(exports, "aria_colcount", ()=>aria_colcount);
/**
 * The aria-colindex attribute defines an element's column index or position with
 * respect to the total number of columns within a table, grid, or treegrid.
 */ parcelHelpers.export(exports, "aria_colindex", ()=>aria_colindex);
/**
 * The aria-colindextext attribute defines a human-readable text alternative of
 * the numeric aria-colindex.
 */ parcelHelpers.export(exports, "aria_colindextext", ()=>aria_colindextext);
/**
 * The aria-colspan attribute defines the number of columns spanned by a cell
 * or gridcell within a table, grid, or treegrid.
 */ parcelHelpers.export(exports, "aria_colspan", ()=>aria_colspan);
/**
 * The global aria-controls property identifies the element (or elements) whose
 * contents or presence are controlled by the element on which this attribute is
 * set.
 */ parcelHelpers.export(exports, "aria_controls", ()=>aria_controls);
/**
 * A non-null aria-current state on an element indicates that this element represents
 * the current item within a container or set of related elements.
 */ parcelHelpers.export(exports, "aria_current", ()=>aria_current);
/**
 * The global aria-describedby attribute identifies the element (or elements)
 * that describes the element on which the attribute is set.
 */ parcelHelpers.export(exports, "aria_describedby", ()=>aria_describedby);
/**
 * The global aria-description attribute defines a string value that describes
 * or annotates the current element.
 */ parcelHelpers.export(exports, "aria_description", ()=>aria_description);
/**
 * The global aria-details attribute identifies the element (or elements) that
 * provide additional information related to the object.
 */ parcelHelpers.export(exports, "aria_details", ()=>aria_details);
/**
 * The aria-disabled state indicates that the element is perceivable but disabled,
 * so it is not editable or otherwise operable.
 */ parcelHelpers.export(exports, "aria_disabled", ()=>aria_disabled);
/**
 * The aria-errormessage attribute on an object identifies the element that
 * provides an error message for that object.
 */ parcelHelpers.export(exports, "aria_errormessage", ()=>aria_errormessage);
/**
 * The aria-expanded attribute is set on an element to indicate if a control is
 * expanded or collapsed, and whether or not the controlled elements are displayed
 * or hidden.
 */ parcelHelpers.export(exports, "aria_expanded", ()=>aria_expanded);
/**
 * The global aria-flowto attribute identifies the next element (or elements) in
 * an alternate reading order of content. This allows assistive technology to
 * override the general default of reading in document source order at the user's
 * discretion.
 */ parcelHelpers.export(exports, "aria_flowto", ()=>aria_flowto);
/**
 * The aria-haspopup attribute indicates the availability and type of interactive
 * popup element that can be triggered by the element on which the attribute is
 * set.
 */ parcelHelpers.export(exports, "aria_haspopup", ()=>aria_haspopup);
/**
 * The aria-hidden state indicates whether the element is exposed to an accessibility
 * API.
 */ parcelHelpers.export(exports, "aria_hidden", ()=>aria_hidden);
/**
 * The aria-invalid state indicates the entered value does not conform to the
 * format expected by the application.
 */ parcelHelpers.export(exports, "aria_invalid", ()=>aria_invalid);
/**
 * The global aria-keyshortcuts attribute indicates keyboard shortcuts that an
 * author has implemented to activate or give focus to an element.
 */ parcelHelpers.export(exports, "aria_keyshortcuts", ()=>aria_keyshortcuts);
/**
 * The aria-label attribute defines a string value that can be used to name an
 * element, as long as the element's role does not prohibit naming.
 */ parcelHelpers.export(exports, "aria_label", ()=>aria_label);
/**
 * The aria-labelledby attribute identifies the element (or elements) that labels
 * the element it is applied to.
 */ parcelHelpers.export(exports, "aria_labelledby", ()=>aria_labelledby);
/**
 * The aria-level attribute defines the hierarchical level of an element within
 * a structure.
 */ parcelHelpers.export(exports, "aria_level", ()=>aria_level);
/**
 * The global aria-live attribute indicates that an element will be updated, and
 * describes the types of updates the user agents, assistive technologies, and
 * user can expect from the live region.
 */ parcelHelpers.export(exports, "aria_live", ()=>aria_live);
/**
 * The aria-modal attribute indicates whether an element is modal when displayed.
 */ parcelHelpers.export(exports, "aria_modal", ()=>aria_modal);
/**
 * The aria-multiline attribute indicates whether a textbox accepts multiple
 * lines of input or only a single line.
 */ parcelHelpers.export(exports, "aria_multiline", ()=>aria_multiline);
/**
 * The aria-multiselectable attribute indicates that the user may select more
 * than one item from the current selectable descendants.
 */ parcelHelpers.export(exports, "aria_multiselectable", ()=>aria_multiselectable);
/**
 * The aria-orientation attribute indicates whether the element's orientation is
 * horizontal, vertical, or unknown/ambiguous.
 */ parcelHelpers.export(exports, "aria_orientation", ()=>aria_orientation);
/**
 * The aria-owns attribute identifies an element (or elements) in order to define
 * a visual, functional, or contextual relationship between a parent and its
 * child elements when the DOM hierarchy cannot be used to represent the relationship.
 */ parcelHelpers.export(exports, "aria_owns", ()=>aria_owns);
/**
 * The aria-placeholder attribute defines a short hint (a word or short phrase)
 * intended to help the user with data entry when a form control has no value.
 * The hint can be a sample value or a brief description of the expected format.
 */ parcelHelpers.export(exports, "aria_placeholder", ()=>aria_placeholder);
/**
 * The aria-posinset attribute defines an element's number or position in the
 * current set of listitems or treeitems when not all items are present in the
 * DOM.
 */ parcelHelpers.export(exports, "aria_posinset", ()=>aria_posinset);
/**
 * The aria-pressed attribute indicates the current "pressed" state of a toggle
 * button.
 */ parcelHelpers.export(exports, "aria_pressed", ()=>aria_pressed);
/**
 * The aria-readonly attribute indicates that the element is not editable, but is
 * otherwise operable.
 */ parcelHelpers.export(exports, "aria_readonly", ()=>aria_readonly);
/**
 * Used in ARIA live regions, the global aria-relevant attribute indicates what
 * notifications the user agent will trigger when the accessibility tree within
 * a live region is modified.
 */ parcelHelpers.export(exports, "aria_relevant", ()=>aria_relevant);
/**
 * The aria-required attribute indicates that user input is required on the element
 * before a form may be submitted.
 */ parcelHelpers.export(exports, "aria_required", ()=>aria_required);
/**
 * The aria-roledescription attribute defines a human-readable, author-localised
 * description for the role of an element.
 */ parcelHelpers.export(exports, "aria_roledescription", ()=>aria_roledescription);
/**
 * The aria-rowcount attribute defines the total number of rows in a table,
 * grid, or treegrid.
 */ parcelHelpers.export(exports, "aria_rowcount", ()=>aria_rowcount);
/**
 * The aria-rowindex attribute defines an element's position with respect to the
 * total number of rows within a table, grid, or treegrid.
 */ parcelHelpers.export(exports, "aria_rowindex", ()=>aria_rowindex);
/**
 * The aria-rowindextext attribute defines a human-readable text alternative of
 * aria-rowindex.
 */ parcelHelpers.export(exports, "aria_rowindextext", ()=>aria_rowindextext);
/**
 * The aria-rowspan attribute defines the number of rows spanned by a cell or
 * gridcell within a table, grid, or treegrid.
 */ parcelHelpers.export(exports, "aria_rowspan", ()=>aria_rowspan);
/**
 * The aria-selected attribute indicates the current "selected" state of various
 * widgets.
 */ parcelHelpers.export(exports, "aria_selected", ()=>aria_selected);
/**
 * The aria-setsize attribute defines the number of items in the current set of
 * listitems or treeitems when not all items in the set are present in the DOM.
 */ parcelHelpers.export(exports, "aria_setsize", ()=>aria_setsize);
/**
 * The aria-sort attribute indicates if items in a table or grid are sorted in
 * ascending or descending order.
 */ parcelHelpers.export(exports, "aria_sort", ()=>aria_sort);
/**
 * The aria-valuemax attribute defines the maximum allowed value for a range
 * widget.
 */ parcelHelpers.export(exports, "aria_valuemax", ()=>aria_valuemax);
/**
 * The aria-valuemin attribute defines the minimum allowed value for a range
 * widget.
 */ parcelHelpers.export(exports, "aria_valuemin", ()=>aria_valuemin);
/**
 * The aria-valuenow attribute defines the current value for a range widget.
 */ parcelHelpers.export(exports, "aria_valuenow", ()=>aria_valuenow);
/**
 * The aria-valuetext attribute defines the human-readable text alternative of
 * aria-valuenow for a range widget.
 */ parcelHelpers.export(exports, "aria_valuetext", ()=>aria_valuetext);
var _jsonMjs = require("../../gleam_json/gleam/json.mjs");
var _intMjs = require("../../gleam_stdlib/gleam/int.mjs");
var _stringMjs = require("../../gleam_stdlib/gleam/string.mjs");
var _gleamMjs = require("../gleam.mjs");
var _vattrMjs = require("../lustre/vdom/vattr.mjs");
function attribute(name, value) {
    return _vattrMjs.attribute(name, value);
}
function property(name, value) {
    return _vattrMjs.property(name, value);
}
function boolean_attribute(name, value) {
    if (value) return attribute(name, "");
    else return property(name, _jsonMjs.bool(false));
}
function accesskey(key) {
    return attribute("accesskey", key);
}
function autocapitalize(value) {
    return attribute("autocapitalize", value);
}
function autocorrect(enabled) {
    return boolean_attribute("autocorrect", enabled);
}
function autofocus(should_autofocus) {
    return boolean_attribute("autofocus", should_autofocus);
}
function class$(name) {
    return attribute("class", name);
}
function none() {
    return class$("");
}
function do_classes(loop$names, loop$class) {
    while(true){
        let names = loop$names;
        let class$ = loop$class;
        if (names instanceof (0, _gleamMjs.Empty)) return class$;
        else {
            let $ = names.head[1];
            if ($) {
                let rest = names.tail;
                let name$1 = names.head[0];
                return class$ + name$1 + " " + do_classes(rest, class$);
            } else {
                let rest = names.tail;
                loop$names = rest;
                loop$class = class$;
            }
        }
    }
}
function classes(names) {
    return class$(do_classes(names, ""));
}
function contenteditable(is_editable) {
    return attribute("contenteditable", is_editable);
}
function data(key, value) {
    return attribute("data-" + key, value);
}
function dir(direction) {
    return attribute("dir", direction);
}
function draggable(is_draggable) {
    return attribute("draggable", (()=>{
        if (is_draggable) return "true";
        else return "false";
    })());
}
function enterkeyhint(value) {
    return attribute("enterkeyhint", value);
}
function hidden(is_hidden) {
    return boolean_attribute("hidden", is_hidden);
}
function id(value) {
    return attribute("id", value);
}
function inert(is_inert) {
    return boolean_attribute("inert", is_inert);
}
function inputmode(value) {
    return attribute("inputmode", value);
}
function is(value) {
    return attribute("is", value);
}
function itemid(id) {
    return attribute("itemid", id);
}
function itemprop(name) {
    return attribute("itemprop", name);
}
function itemscope(has_scope) {
    return boolean_attribute("itemscope", has_scope);
}
function itemtype(url) {
    return attribute("itemtype", url);
}
function lang(language) {
    return attribute("lang", language);
}
function nonce(value) {
    return attribute("nonce", value);
}
function popover(value) {
    return attribute("popover", value);
}
function spellcheck(should_check) {
    return attribute("spellcheck", (()=>{
        if (should_check) return "true";
        else return "false";
    })());
}
function style(property, value) {
    if (property === "") return class$("");
    else if (value === "") return class$("");
    else return attribute("style", property + ":" + value + ";");
}
function do_styles(loop$properties, loop$styles) {
    while(true){
        let properties = loop$properties;
        let styles = loop$styles;
        if (properties instanceof (0, _gleamMjs.Empty)) return styles;
        else {
            let $ = properties.head[0];
            if ($ === "") {
                let rest = properties.tail;
                loop$properties = rest;
                loop$styles = styles;
            } else {
                let $1 = properties.head[1];
                if ($1 === "") {
                    let rest = properties.tail;
                    loop$properties = rest;
                    loop$styles = styles;
                } else {
                    let rest = properties.tail;
                    let name$1 = $;
                    let value$1 = $1;
                    loop$properties = rest;
                    loop$styles = styles + name$1 + ":" + value$1 + ";";
                }
            }
        }
    }
}
function styles(properties) {
    return attribute("style", do_styles(properties, ""));
}
function tabindex(index) {
    return attribute("tabindex", _intMjs.to_string(index));
}
function title(text) {
    return attribute("title", text);
}
function translate(should_translate) {
    return attribute("translate", (()=>{
        if (should_translate) return "yes";
        else return "no";
    })());
}
function writingsuggestions(enabled) {
    return attribute("writingsuggestions", (()=>{
        if (enabled) return "true";
        else return "false";
    })());
}
function open(is_open) {
    return boolean_attribute("open", is_open);
}
function href(url) {
    return attribute("href", url);
}
function target(value) {
    return attribute("target", value);
}
function download(filename) {
    return attribute("download", filename);
}
function ping(urls) {
    return attribute("ping", _stringMjs.join(urls, " "));
}
function rel(value) {
    return attribute("rel", value);
}
function hreflang(language) {
    return attribute("hreflang", language);
}
function referrerpolicy(value) {
    return attribute("referrerpolicy", value);
}
function as_(value) {
    return attribute("as", value);
}
function blocking(value) {
    return attribute("blocking", (()=>{
        if (value) return "render";
        else return "";
    })());
}
function integrity(hash) {
    return attribute("integrity", hash);
}
function alt(text) {
    return attribute("alt", text);
}
function src(url) {
    return attribute("src", url);
}
function srcset(sources) {
    return attribute("srcset", sources);
}
function sizes(value) {
    return attribute("sizes", value);
}
function crossorigin(value) {
    return attribute("crossorigin", value);
}
function usemap(value) {
    return attribute("usemap", value);
}
function ismap(is_map) {
    return boolean_attribute("ismap", is_map);
}
function width(value) {
    return attribute("width", _intMjs.to_string(value));
}
function height(value) {
    return attribute("height", _intMjs.to_string(value));
}
function decoding(value) {
    return attribute("decoding", value);
}
function loading(value) {
    return attribute("loading", value);
}
function fetchpriority(value) {
    return attribute("fetchpriority", value);
}
function accept_charset(charsets) {
    return attribute("accept-charset", charsets);
}
function action(url) {
    return attribute("action", url);
}
function enctype(encoding_type) {
    return attribute("enctype", encoding_type);
}
function method(http_method) {
    return attribute("method", http_method);
}
function novalidate(disable_validation) {
    return boolean_attribute("novalidate", disable_validation);
}
function accept(values) {
    return attribute("accept", _stringMjs.join(values, ","));
}
function alpha(allowed) {
    return boolean_attribute("alpha", allowed);
}
function autocomplete(value) {
    return attribute("autocomplete", value);
}
function checked(is_checked) {
    return boolean_attribute("checked", is_checked);
}
function default_checked(is_checked) {
    return boolean_attribute("virtual:defaultChecked", is_checked);
}
function colorspace(value) {
    return attribute("colorspace", value);
}
function cols(value) {
    return attribute("cols", _intMjs.to_string(value));
}
function dirname(direction) {
    return attribute("dirname", direction);
}
function disabled(is_disabled) {
    return boolean_attribute("disabled", is_disabled);
}
function for$(id) {
    return attribute("for", id);
}
function form(id) {
    return attribute("form", id);
}
function formaction(url) {
    return attribute("formaction", url);
}
function formenctype(encoding_type) {
    return attribute("formenctype", encoding_type);
}
function formmethod(method) {
    return attribute("formmethod", method);
}
function formnovalidate(no_validate) {
    return boolean_attribute("formnovalidate", no_validate);
}
function formtarget(target) {
    return attribute("formtarget", target);
}
function list(id) {
    return attribute("list", id);
}
function max(value) {
    return attribute("max", value);
}
function maxlength(length) {
    return attribute("maxlength", _intMjs.to_string(length));
}
function min(value) {
    return attribute("min", value);
}
function minlength(length) {
    return attribute("minlength", _intMjs.to_string(length));
}
function multiple(allow_multiple) {
    return boolean_attribute("multiple", allow_multiple);
}
function name(element_name) {
    return attribute("name", element_name);
}
function pattern(regex) {
    return attribute("pattern", regex);
}
function placeholder(text) {
    return attribute("placeholder", text);
}
function popovertarget(id) {
    return attribute("popovertarget", id);
}
function popovertargetaction(action) {
    return attribute("popovertargetaction", action);
}
function readonly(is_readonly) {
    return boolean_attribute("readonly", is_readonly);
}
function required(is_required) {
    return boolean_attribute("required", is_required);
}
function rows(value) {
    return attribute("rows", _intMjs.to_string(value));
}
function selected(is_selected) {
    return boolean_attribute("selected", is_selected);
}
function default_selected(is_selected) {
    return boolean_attribute("virtual:defaultSelected", is_selected);
}
function size(value) {
    return attribute("size", value);
}
function step(value) {
    return attribute("step", value);
}
function type_(control_type) {
    return attribute("type", control_type);
}
function value(control_value) {
    return attribute("value", control_value);
}
function default_value(control_value) {
    return attribute("virtual:defaultValue", control_value);
}
function http_equiv(value) {
    return attribute("http-equiv", value);
}
function content(value) {
    return attribute("content", value);
}
function charset(value) {
    return attribute("charset", value);
}
function media(query) {
    return attribute("media", query);
}
function autoplay(auto_play) {
    return boolean_attribute("autoplay", auto_play);
}
function controls(show_controls) {
    return boolean_attribute("controls", show_controls);
}
function loop(should_loop) {
    return boolean_attribute("loop", should_loop);
}
function muted(is_muted) {
    return boolean_attribute("muted", is_muted);
}
function playsinline(play_inline) {
    return boolean_attribute("playsinline", play_inline);
}
function poster(url) {
    return attribute("poster", url);
}
function preload(value) {
    return attribute("preload", value);
}
function shadowrootmode(mode) {
    return attribute("shadowrootmode", mode);
}
function shadowrootdelegatesfocus(delegates) {
    return boolean_attribute("shadowrootdelegatesfocus", delegates);
}
function shadowrootclonable(clonable) {
    return boolean_attribute("shadowrootclonable", clonable);
}
function shadowrootserializable(serializable) {
    return boolean_attribute("shadowrootserializable", serializable);
}
function abbr(value) {
    return attribute("abbr", value);
}
function colspan(value) {
    return attribute("colspan", _intMjs.to_string(value));
}
function headers(ids) {
    return attribute("headers", _stringMjs.join(ids, " "));
}
function rowspan(value) {
    return attribute("rowspan", (()=>{
        let _pipe = value;
        let _pipe$1 = _intMjs.max(_pipe, 0);
        let _pipe$2 = _intMjs.min(_pipe$1, 65534);
        return _intMjs.to_string(_pipe$2);
    })());
}
function span(value) {
    return attribute("span", _intMjs.to_string(value));
}
function scope(value) {
    return attribute("scope", value);
}
function datetime(value) {
    return attribute("datetime", value);
}
function aria(name, value) {
    return attribute("aria-" + name, value);
}
function role(name) {
    return attribute("role", name);
}
function aria_activedescendant(id) {
    return aria("activedescendant", id);
}
function aria_atomic(value) {
    return aria("atomic", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_autocomplete(value) {
    return aria("autocomplete", value);
}
function aria_braillelabel(value) {
    return aria("braillelabel", value);
}
function aria_brailleroledescription(value) {
    return aria("brailleroledescription", value);
}
function aria_busy(value) {
    return aria("busy", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_checked(value) {
    return aria("checked", value);
}
function aria_colcount(value) {
    return aria("colcount", _intMjs.to_string(value));
}
function aria_colindex(value) {
    return aria("colindex", _intMjs.to_string(value));
}
function aria_colindextext(value) {
    return aria("colindextext", value);
}
function aria_colspan(value) {
    return aria("colspan", _intMjs.to_string(value));
}
function aria_controls(value) {
    return aria("controls", value);
}
function aria_current(value) {
    return aria("current", value);
}
function aria_describedby(value) {
    return aria("describedby", value);
}
function aria_description(value) {
    return aria("description", value);
}
function aria_details(value) {
    return aria("details", value);
}
function aria_disabled(value) {
    return aria("disabled", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_errormessage(value) {
    return aria("errormessage", value);
}
function aria_expanded(value) {
    return aria("expanded", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_flowto(value) {
    return aria("flowto", value);
}
function aria_haspopup(value) {
    return aria("haspopup", value);
}
function aria_hidden(value) {
    return aria("hidden", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_invalid(value) {
    return aria("invalid", value);
}
function aria_keyshortcuts(value) {
    return aria("keyshortcuts", value);
}
function aria_label(value) {
    return aria("label", value);
}
function aria_labelledby(value) {
    return aria("labelledby", value);
}
function aria_level(value) {
    return aria("level", _intMjs.to_string(value));
}
function aria_live(value) {
    return aria("live", value);
}
function aria_modal(value) {
    return aria("modal", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_multiline(value) {
    return aria("multiline", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_multiselectable(value) {
    return aria("multiselectable", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_orientation(value) {
    return aria("orientation", value);
}
function aria_owns(value) {
    return aria("owns", value);
}
function aria_placeholder(value) {
    return aria("placeholder", value);
}
function aria_posinset(value) {
    return aria("posinset", _intMjs.to_string(value));
}
function aria_pressed(value) {
    return aria("pressed", value);
}
function aria_readonly(value) {
    return aria("readonly", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_relevant(value) {
    return aria("relevant", value);
}
function aria_required(value) {
    return aria("required", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_roledescription(value) {
    return aria("roledescription", value);
}
function aria_rowcount(value) {
    return aria("rowcount", _intMjs.to_string(value));
}
function aria_rowindex(value) {
    return aria("rowindex", _intMjs.to_string(value));
}
function aria_rowindextext(value) {
    return aria("rowindextext", value);
}
function aria_rowspan(value) {
    return aria("rowspan", _intMjs.to_string(value));
}
function aria_selected(value) {
    return aria("selected", (()=>{
        if (value) return "true";
        else return "false";
    })());
}
function aria_setsize(value) {
    return aria("setsize", _intMjs.to_string(value));
}
function aria_sort(value) {
    return aria("sort", value);
}
function aria_valuemax(value) {
    return aria("valuemax", value);
}
function aria_valuemin(value) {
    return aria("valuemin", value);
}
function aria_valuenow(value) {
    return aria("valuenow", value);
}
function aria_valuetext(value) {
    return aria("valuetext", value);
}

},{"../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam_stdlib/gleam/int.mjs":"32hLf","../../gleam_stdlib/gleam/string.mjs":"aB8qb","../gleam.mjs":"jNPQG","../lustre/vdom/vattr.mjs":"jrrcC","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"8Pq32":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "to_string_tree", ()=>(0, _gleamJsonFfiMjs.json_to_string));
parcelHelpers.export(exports, "UnexpectedEndOfInput", ()=>UnexpectedEndOfInput);
parcelHelpers.export(exports, "DecodeError$UnexpectedEndOfInput", ()=>DecodeError$UnexpectedEndOfInput);
parcelHelpers.export(exports, "DecodeError$isUnexpectedEndOfInput", ()=>DecodeError$isUnexpectedEndOfInput);
parcelHelpers.export(exports, "UnexpectedByte", ()=>UnexpectedByte);
parcelHelpers.export(exports, "DecodeError$UnexpectedByte", ()=>DecodeError$UnexpectedByte);
parcelHelpers.export(exports, "DecodeError$isUnexpectedByte", ()=>DecodeError$isUnexpectedByte);
parcelHelpers.export(exports, "DecodeError$UnexpectedByte$0", ()=>DecodeError$UnexpectedByte$0);
parcelHelpers.export(exports, "UnexpectedSequence", ()=>UnexpectedSequence);
parcelHelpers.export(exports, "DecodeError$UnexpectedSequence", ()=>DecodeError$UnexpectedSequence);
parcelHelpers.export(exports, "DecodeError$isUnexpectedSequence", ()=>DecodeError$isUnexpectedSequence);
parcelHelpers.export(exports, "DecodeError$UnexpectedSequence$0", ()=>DecodeError$UnexpectedSequence$0);
parcelHelpers.export(exports, "UnableToDecode", ()=>UnableToDecode);
parcelHelpers.export(exports, "DecodeError$UnableToDecode", ()=>DecodeError$UnableToDecode);
parcelHelpers.export(exports, "DecodeError$isUnableToDecode", ()=>DecodeError$isUnableToDecode);
parcelHelpers.export(exports, "DecodeError$UnableToDecode$0", ()=>DecodeError$UnableToDecode$0);
/**
 * Decode a JSON string into dynamically typed data which can be decoded into
 * typed data with the `gleam/dynamic` module.
 *
 * ## Examples
 *
 * ```gleam
 * > parse("[1,2,3]", decode.list(of: decode.int))
 * Ok([1, 2, 3])
 * ```
 *
 * ```gleam
 * > parse("[", decode.list(of: decode.int))
 * Error(UnexpectedEndOfInput)
 * ```
 *
 * ```gleam
 * > parse("1", decode.string)
 * Error(UnableToDecode([decode.DecodeError("String", "Int", [])]))
 * ```
 */ parcelHelpers.export(exports, "parse", ()=>parse);
/**
 * Decode a JSON bit string into dynamically typed data which can be decoded
 * into typed data with the `gleam/dynamic` module.
 *
 * ## Examples
 *
 * ```gleam
 * > parse_bits(<<"[1,2,3]">>, decode.list(of: decode.int))
 * Ok([1, 2, 3])
 * ```
 *
 * ```gleam
 * > parse_bits(<<"[">>, decode.list(of: decode.int))
 * Error(UnexpectedEndOfInput)
 * ```
 *
 * ```gleam
 * > parse_bits(<<"1">>, decode.string)
 * Error(UnableToDecode([decode.DecodeError("String", "Int", [])])),
 * ```
 */ parcelHelpers.export(exports, "parse_bits", ()=>parse_bits);
/**
 * Convert a JSON value into a string.
 *
 * Where possible prefer the `to_string_tree` function as it is faster than
 * this function, and BEAM VM IO is optimised for sending `StringTree` data.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(array([1, 2, 3], of: int))
 * "[1,2,3]"
 * ```
 */ parcelHelpers.export(exports, "to_string", ()=>to_string);
/**
 * Encode a string into JSON, using normal JSON escaping.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(string("Hello!"))
 * "\"Hello!\""
 * ```
 */ parcelHelpers.export(exports, "string", ()=>string);
/**
 * Encode a bool into JSON.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(bool(False))
 * "false"
 * ```
 */ parcelHelpers.export(exports, "bool", ()=>bool);
/**
 * Encode an int into JSON.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(int(50))
 * "50"
 * ```
 */ parcelHelpers.export(exports, "int", ()=>int);
/**
 * Encode a float into JSON.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(float(4.7))
 * "4.7"
 * ```
 */ parcelHelpers.export(exports, "float", ()=>float);
/**
 * The JSON value null.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(null())
 * "null"
 * ```
 */ parcelHelpers.export(exports, "null$", ()=>null$);
/**
 * Encode an optional value into JSON, using null if it is the `None` variant.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(nullable(Some(50), of: int))
 * "50"
 * ```
 *
 * ```gleam
 * > to_string(nullable(None, of: int))
 * "null"
 * ```
 */ parcelHelpers.export(exports, "nullable", ()=>nullable);
/**
 * Encode a list of key-value pairs into a JSON object.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(object([
 *   #("game", string("Pac-Man")),
 *   #("score", int(3333360)),
 * ]))
 * "{\"game\":\"Pac-Mac\",\"score\":3333360}"
 * ```
 */ parcelHelpers.export(exports, "object", ()=>object);
/**
 * Encode a list of JSON values into a JSON array.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(preprocessed_array([int(1), float(2.0), string("3")]))
 * "[1, 2.0, \"3\"]"
 * ```
 */ parcelHelpers.export(exports, "preprocessed_array", ()=>preprocessed_array);
/**
 * Encode a list into a JSON array.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(array([1, 2, 3], of: int))
 * "[1, 2, 3]"
 * ```
 */ parcelHelpers.export(exports, "array", ()=>array);
/**
 * Encode a Dict into a JSON object using the supplied functions to encode
 * the keys and the values respectively.
 *
 * ## Examples
 *
 * ```gleam
 * > to_string(dict(dict.from_list([ #(3, 3.0), #(4, 4.0)]), int.to_string, float)
 * "{\"3\": 3.0, \"4\": 4.0}"
 * ```
 */ parcelHelpers.export(exports, "dict", ()=>dict);
var _bitArrayMjs = require("../../gleam_stdlib/gleam/bit_array.mjs");
var _dictMjs = require("../../gleam_stdlib/gleam/dict.mjs");
var _dynamicMjs = require("../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _listMjs = require("../../gleam_stdlib/gleam/list.mjs");
var _optionMjs = require("../../gleam_stdlib/gleam/option.mjs");
var _resultMjs = require("../../gleam_stdlib/gleam/result.mjs");
var _stringTreeMjs = require("../../gleam_stdlib/gleam/string_tree.mjs");
var _gleamMjs = require("../gleam.mjs");
var _gleamJsonFfiMjs = require("../gleam_json_ffi.mjs");
class UnexpectedEndOfInput extends (0, _gleamMjs.CustomType) {
}
const DecodeError$UnexpectedEndOfInput = ()=>new UnexpectedEndOfInput();
const DecodeError$isUnexpectedEndOfInput = (value)=>value instanceof UnexpectedEndOfInput;
class UnexpectedByte extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const DecodeError$UnexpectedByte = ($0)=>new UnexpectedByte($0);
const DecodeError$isUnexpectedByte = (value)=>value instanceof UnexpectedByte;
const DecodeError$UnexpectedByte$0 = (value)=>value[0];
class UnexpectedSequence extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const DecodeError$UnexpectedSequence = ($0)=>new UnexpectedSequence($0);
const DecodeError$isUnexpectedSequence = (value)=>value instanceof UnexpectedSequence;
const DecodeError$UnexpectedSequence$0 = (value)=>value[0];
class UnableToDecode extends (0, _gleamMjs.CustomType) {
    constructor($0){
        super();
        this[0] = $0;
    }
}
const DecodeError$UnableToDecode = ($0)=>new UnableToDecode($0);
const DecodeError$isUnableToDecode = (value)=>value instanceof UnableToDecode;
const DecodeError$UnableToDecode$0 = (value)=>value[0];
function do_parse(json, decoder) {
    return _resultMjs.try$((0, _gleamJsonFfiMjs.decode)(json), (dynamic_value)=>{
        let _pipe = _decodeMjs.run(dynamic_value, decoder);
        return _resultMjs.map_error(_pipe, (var0)=>{
            return new UnableToDecode(var0);
        });
    });
}
function parse(json, decoder) {
    return do_parse(json, decoder);
}
function decode_to_dynamic(json) {
    let $ = _bitArrayMjs.to_string(json);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let string$1 = $[0];
        return (0, _gleamJsonFfiMjs.decode)(string$1);
    } else return new (0, _gleamMjs.Error)(new UnexpectedByte(""));
}
function parse_bits(json, decoder) {
    return _resultMjs.try$(decode_to_dynamic(json), (dynamic_value)=>{
        let _pipe = _decodeMjs.run(dynamic_value, decoder);
        return _resultMjs.map_error(_pipe, (var0)=>{
            return new UnableToDecode(var0);
        });
    });
}
function to_string(json) {
    return (0, _gleamJsonFfiMjs.json_to_string)(json);
}
function string(input) {
    return (0, _gleamJsonFfiMjs.identity)(input);
}
function bool(input) {
    return (0, _gleamJsonFfiMjs.identity)(input);
}
function int(input) {
    return (0, _gleamJsonFfiMjs.identity)(input);
}
function float(input) {
    return (0, _gleamJsonFfiMjs.identity)(input);
}
function null$() {
    return (0, _gleamJsonFfiMjs.do_null)();
}
function nullable(input, inner_type) {
    if (input instanceof (0, _optionMjs.Some)) {
        let value = input[0];
        return inner_type(value);
    } else return null$();
}
function object(entries) {
    return (0, _gleamJsonFfiMjs.object)(entries);
}
function preprocessed_array(from) {
    return (0, _gleamJsonFfiMjs.array)(from);
}
function array(entries, inner_type) {
    let _pipe = entries;
    let _pipe$1 = _listMjs.map(_pipe, inner_type);
    return preprocessed_array(_pipe$1);
}
function dict(dict, keys, values) {
    return object(_dictMjs.fold(dict, (0, _gleamMjs.toList)([]), (acc, k, v)=>{
        return (0, _gleamMjs.prepend)([
            keys(k),
            values(v)
        ], acc);
    }));
}

},{"../../gleam_stdlib/gleam/bit_array.mjs":"69HLR","../../gleam_stdlib/gleam/dict.mjs":"b8yrU","../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../gleam_stdlib/gleam/result.mjs":"oBmFG","../../gleam_stdlib/gleam/string_tree.mjs":"8IH0o","../gleam.mjs":"bhKz9","../gleam_json_ffi.mjs":"5DYj0","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"bhKz9":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5DYj0":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "json_to_string", ()=>json_to_string);
parcelHelpers.export(exports, "object", ()=>object);
parcelHelpers.export(exports, "identity", ()=>identity);
parcelHelpers.export(exports, "array", ()=>array);
parcelHelpers.export(exports, "do_null", ()=>do_null);
parcelHelpers.export(exports, "decode", ()=>decode);
parcelHelpers.export(exports, "getJsonDecodeError", ()=>getJsonDecodeError);
var _gleamMjs = require("./gleam.mjs");
var _jsonMjs = require("./gleam/json.mjs");
function json_to_string(json) {
    return JSON.stringify(json);
}
function object(entries) {
    return Object.fromEntries(entries);
}
function identity(x) {
    return x;
}
function array(list) {
    const array = [];
    while((0, _gleamMjs.List$isNonEmpty)(list)){
        array.push((0, _gleamMjs.List$NonEmpty$first)(list));
        list = (0, _gleamMjs.List$NonEmpty$rest)(list);
    }
    return array;
}
function do_null() {
    return null;
}
function decode(string) {
    try {
        const result = JSON.parse(string);
        return (0, _gleamMjs.Result$Ok)(result);
    } catch (err) {
        return (0, _gleamMjs.Result$Error)(getJsonDecodeError(err, string));
    }
}
function getJsonDecodeError(stdErr, json) {
    if (isUnexpectedEndOfInput(stdErr)) return (0, _jsonMjs.DecodeError$UnexpectedEndOfInput)();
    return toUnexpectedByteError(stdErr, json);
}
/**
 * Matches unexpected end of input messages in:
 * - Chromium (edge, chrome, node)
 * - Spidermonkey (firefox)
 * - JavascriptCore (safari)
 *
 * Note that Spidermonkey and JavascriptCore will both incorrectly report some
 * UnexpectedByte errors as UnexpectedEndOfInput errors. For example:
 *
 * @example
 * // in JavascriptCore
 * JSON.parse('{"a"]: "b"})
 * // => JSON Parse error: Expected ':' before value
 *
 * JSON.parse('{"a"')
 * // => JSON Parse error: Expected ':' before value
 *
 * // in Chromium (correct)
 * JSON.parse('{"a"]: "b"})
 * // => Unexpected token ] in JSON at position 4
 *
 * JSON.parse('{"a"')
 * // => Unexpected end of JSON input
 */ function isUnexpectedEndOfInput(err) {
    const unexpectedEndOfInputRegex = /((unexpected (end|eof))|(end of data)|(unterminated string)|(json( parse error|\.parse)\: expected '(\:|\}|\])'))/i;
    return unexpectedEndOfInputRegex.test(err.message);
}
/**
 * Converts a SyntaxError to an UnexpectedByte error based on the JS runtime.
 *
 * For Chromium, the unexpected byte and position are reported by the runtime.
 *
 * For JavascriptCore, only the unexpected byte is reported by the runtime, so
 * there is no way to know which position that character is in unless we then
 * parse the string again ourselves. So instead, the position is reported as 0.
 *
 * For Spidermonkey, the position is reported by the runtime as a line and column number
 * and the unexpected byte is found using those coordinates.
 */ function toUnexpectedByteError(err, json) {
    let converters = [
        v8UnexpectedByteError,
        oldV8UnexpectedByteError,
        jsCoreUnexpectedByteError,
        spidermonkeyUnexpectedByteError
    ];
    for (let converter of converters){
        let result = converter(err, json);
        if (result) return result;
    }
    return (0, _jsonMjs.DecodeError$UnexpectedByte)("");
}
/**
 * Matches unexpected byte messages in:
 * - V8 (edge, chrome, node)
 *
 * Matches the character but not the position as this is no longer reported by
 * V8. Boo!
 */ function v8UnexpectedByteError(err) {
    const regex = /unexpected token '(.)', ".+" is not valid JSON/i;
    const match = regex.exec(err.message);
    if (!match) return null;
    const byte = toHex(match[1]);
    return (0, _jsonMjs.DecodeError$UnexpectedByte)(byte);
}
/**
 * Matches unexpected byte messages in:
 * - V8 (edge, chrome, node)
 *
 * No longer works in current versions of V8.
 *
 * Matches the character and its position.
 */ function oldV8UnexpectedByteError(err) {
    const regex = /unexpected token (.) in JSON at position (\d+)/i;
    const match = regex.exec(err.message);
    if (!match) return null;
    const byte = toHex(match[1]);
    return (0, _jsonMjs.DecodeError$UnexpectedByte)(byte);
}
/**
 * Matches unexpected byte messages in:
 * - Spidermonkey (firefox)
 *
 * Matches the position in a 2d grid only and not the character.
 */ function spidermonkeyUnexpectedByteError(err, json) {
    const regex = /(unexpected character|expected .*) at line (\d+) column (\d+)/i;
    const match = regex.exec(err.message);
    if (!match) return null;
    const line = Number(match[2]);
    const column = Number(match[3]);
    const position = getPositionFromMultiline(line, column, json);
    const byte = toHex(json[position]);
    return (0, _jsonMjs.DecodeError$UnexpectedByte)(byte);
}
/**
 * Matches unexpected byte messages in:
 * - JavascriptCore (safari)
 *
 * JavascriptCore only reports what the character is and not its position.
 */ function jsCoreUnexpectedByteError(err) {
    const regex = /unexpected (identifier|token) "(.)"/i;
    const match = regex.exec(err.message);
    if (!match) return null;
    const byte = toHex(match[2]);
    return (0, _jsonMjs.DecodeError$UnexpectedByte)(byte);
}
function toHex(char) {
    return "0x" + char.charCodeAt(0).toString(16).toUpperCase();
}
/**
 * Gets the position of a character in a flattened (i.e. single line) string
 * from a line and column number. Note that the position is 0-indexed and
 * the line and column numbers are 1-indexed.
 *
 * @param {number} line
 * @param {number} column
 * @param {string} string
 */ function getPositionFromMultiline(line, column, string) {
    if (line === 1) return column - 1;
    let currentLn = 1;
    let position = 0;
    string.split("").find((char, idx)=>{
        if (char === "\n") currentLn += 1;
        if (currentLn === line) {
            position = idx + column;
            return true;
        }
        return false;
    });
    return position;
}

},{"./gleam.mjs":"bhKz9","./gleam/json.mjs":"8Pq32","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jrrcC":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "compare", ()=>(0, _vattrFfiMjs.compare));
parcelHelpers.export(exports, "Attribute", ()=>Attribute);
parcelHelpers.export(exports, "Attribute$Attribute", ()=>Attribute$Attribute);
parcelHelpers.export(exports, "Attribute$isAttribute", ()=>Attribute$isAttribute);
parcelHelpers.export(exports, "Attribute$Attribute$kind", ()=>Attribute$Attribute$kind);
parcelHelpers.export(exports, "Attribute$Attribute$0", ()=>Attribute$Attribute$0);
parcelHelpers.export(exports, "Attribute$Attribute$name", ()=>Attribute$Attribute$name);
parcelHelpers.export(exports, "Attribute$Attribute$1", ()=>Attribute$Attribute$1);
parcelHelpers.export(exports, "Attribute$Attribute$value", ()=>Attribute$Attribute$value);
parcelHelpers.export(exports, "Attribute$Attribute$2", ()=>Attribute$Attribute$2);
parcelHelpers.export(exports, "Property", ()=>Property);
parcelHelpers.export(exports, "Attribute$Property", ()=>Attribute$Property);
parcelHelpers.export(exports, "Attribute$isProperty", ()=>Attribute$isProperty);
parcelHelpers.export(exports, "Attribute$Property$kind", ()=>Attribute$Property$kind);
parcelHelpers.export(exports, "Attribute$Property$0", ()=>Attribute$Property$0);
parcelHelpers.export(exports, "Attribute$Property$name", ()=>Attribute$Property$name);
parcelHelpers.export(exports, "Attribute$Property$1", ()=>Attribute$Property$1);
parcelHelpers.export(exports, "Attribute$Property$value", ()=>Attribute$Property$value);
parcelHelpers.export(exports, "Attribute$Property$2", ()=>Attribute$Property$2);
parcelHelpers.export(exports, "Event", ()=>Event);
parcelHelpers.export(exports, "Attribute$Event", ()=>Attribute$Event);
parcelHelpers.export(exports, "Attribute$isEvent", ()=>Attribute$isEvent);
parcelHelpers.export(exports, "Attribute$Event$kind", ()=>Attribute$Event$kind);
parcelHelpers.export(exports, "Attribute$Event$0", ()=>Attribute$Event$0);
parcelHelpers.export(exports, "Attribute$Event$name", ()=>Attribute$Event$name);
parcelHelpers.export(exports, "Attribute$Event$1", ()=>Attribute$Event$1);
parcelHelpers.export(exports, "Attribute$Event$handler", ()=>Attribute$Event$handler);
parcelHelpers.export(exports, "Attribute$Event$2", ()=>Attribute$Event$2);
parcelHelpers.export(exports, "Attribute$Event$include", ()=>Attribute$Event$include);
parcelHelpers.export(exports, "Attribute$Event$3", ()=>Attribute$Event$3);
parcelHelpers.export(exports, "Attribute$Event$prevent_default", ()=>Attribute$Event$prevent_default);
parcelHelpers.export(exports, "Attribute$Event$4", ()=>Attribute$Event$4);
parcelHelpers.export(exports, "Attribute$Event$stop_propagation", ()=>Attribute$Event$stop_propagation);
parcelHelpers.export(exports, "Attribute$Event$5", ()=>Attribute$Event$5);
parcelHelpers.export(exports, "Attribute$Event$debounce", ()=>Attribute$Event$debounce);
parcelHelpers.export(exports, "Attribute$Event$6", ()=>Attribute$Event$6);
parcelHelpers.export(exports, "Attribute$Event$throttle", ()=>Attribute$Event$throttle);
parcelHelpers.export(exports, "Attribute$Event$7", ()=>Attribute$Event$7);
parcelHelpers.export(exports, "Attribute$kind", ()=>Attribute$kind);
parcelHelpers.export(exports, "Attribute$name", ()=>Attribute$name);
parcelHelpers.export(exports, "Handler", ()=>Handler);
parcelHelpers.export(exports, "Handler$Handler", ()=>Handler$Handler);
parcelHelpers.export(exports, "Handler$isHandler", ()=>Handler$isHandler);
parcelHelpers.export(exports, "Handler$Handler$prevent_default", ()=>Handler$Handler$prevent_default);
parcelHelpers.export(exports, "Handler$Handler$0", ()=>Handler$Handler$0);
parcelHelpers.export(exports, "Handler$Handler$stop_propagation", ()=>Handler$Handler$stop_propagation);
parcelHelpers.export(exports, "Handler$Handler$1", ()=>Handler$Handler$1);
parcelHelpers.export(exports, "Handler$Handler$message", ()=>Handler$Handler$message);
parcelHelpers.export(exports, "Handler$Handler$2", ()=>Handler$Handler$2);
parcelHelpers.export(exports, "Never", ()=>Never);
parcelHelpers.export(exports, "EventBehaviour$Never", ()=>EventBehaviour$Never);
parcelHelpers.export(exports, "EventBehaviour$isNever", ()=>EventBehaviour$isNever);
parcelHelpers.export(exports, "EventBehaviour$Never$kind", ()=>EventBehaviour$Never$kind);
parcelHelpers.export(exports, "EventBehaviour$Never$0", ()=>EventBehaviour$Never$0);
parcelHelpers.export(exports, "Possible", ()=>Possible);
parcelHelpers.export(exports, "EventBehaviour$Possible", ()=>EventBehaviour$Possible);
parcelHelpers.export(exports, "EventBehaviour$isPossible", ()=>EventBehaviour$isPossible);
parcelHelpers.export(exports, "EventBehaviour$Possible$kind", ()=>EventBehaviour$Possible$kind);
parcelHelpers.export(exports, "EventBehaviour$Possible$0", ()=>EventBehaviour$Possible$0);
parcelHelpers.export(exports, "Always", ()=>Always);
parcelHelpers.export(exports, "EventBehaviour$Always", ()=>EventBehaviour$Always);
parcelHelpers.export(exports, "EventBehaviour$isAlways", ()=>EventBehaviour$isAlways);
parcelHelpers.export(exports, "EventBehaviour$Always$kind", ()=>EventBehaviour$Always$kind);
parcelHelpers.export(exports, "EventBehaviour$Always$0", ()=>EventBehaviour$Always$0);
parcelHelpers.export(exports, "EventBehaviour$kind", ()=>EventBehaviour$kind);
parcelHelpers.export(exports, "attribute_kind", ()=>attribute_kind);
parcelHelpers.export(exports, "property_kind", ()=>property_kind);
parcelHelpers.export(exports, "event_kind", ()=>event_kind);
parcelHelpers.export(exports, "never_kind", ()=>never_kind);
parcelHelpers.export(exports, "never", ()=>never);
parcelHelpers.export(exports, "possible_kind", ()=>possible_kind);
parcelHelpers.export(exports, "possible", ()=>possible);
parcelHelpers.export(exports, "always_kind", ()=>always_kind);
parcelHelpers.export(exports, "always", ()=>always);
parcelHelpers.export(exports, "merge", ()=>merge);
parcelHelpers.export(exports, "prepare", ()=>prepare);
parcelHelpers.export(exports, "attribute", ()=>attribute);
parcelHelpers.export(exports, "to_string_tree", ()=>to_string_tree);
parcelHelpers.export(exports, "property", ()=>property);
parcelHelpers.export(exports, "event", ()=>event);
parcelHelpers.export(exports, "to_json", ()=>to_json);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
var _orderMjs = require("../../../gleam_stdlib/gleam/order.mjs");
var _stringMjs = require("../../../gleam_stdlib/gleam/string.mjs");
var _stringTreeMjs = require("../../../gleam_stdlib/gleam/string_tree.mjs");
var _houdiniMjs = require("../../../houdini/houdini.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
var _jsonObjectBuilderMjs = require("../../lustre/internals/json_object_builder.mjs");
var _vattrFfiMjs = require("./vattr.ffi.mjs");
class Attribute extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, value){
        super();
        this.kind = kind;
        this.name = name;
        this.value = value;
    }
}
const Attribute$Attribute = (kind, name, value)=>new Attribute(kind, name, value);
const Attribute$isAttribute = (value)=>value instanceof Attribute;
const Attribute$Attribute$kind = (value)=>value.kind;
const Attribute$Attribute$0 = (value)=>value.kind;
const Attribute$Attribute$name = (value)=>value.name;
const Attribute$Attribute$1 = (value)=>value.name;
const Attribute$Attribute$value = (value)=>value.value;
const Attribute$Attribute$2 = (value)=>value.value;
class Property extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, value){
        super();
        this.kind = kind;
        this.name = name;
        this.value = value;
    }
}
const Attribute$Property = (kind, name, value)=>new Property(kind, name, value);
const Attribute$isProperty = (value)=>value instanceof Property;
const Attribute$Property$kind = (value)=>value.kind;
const Attribute$Property$0 = (value)=>value.kind;
const Attribute$Property$name = (value)=>value.name;
const Attribute$Property$1 = (value)=>value.name;
const Attribute$Property$value = (value)=>value.value;
const Attribute$Property$2 = (value)=>value.value;
class Event extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, handler, include, prevent_default, stop_propagation, debounce, throttle){
        super();
        this.kind = kind;
        this.name = name;
        this.handler = handler;
        this.include = include;
        this.prevent_default = prevent_default;
        this.stop_propagation = stop_propagation;
        this.debounce = debounce;
        this.throttle = throttle;
    }
}
const Attribute$Event = (kind, name, handler, include, prevent_default, stop_propagation, debounce, throttle)=>new Event(kind, name, handler, include, prevent_default, stop_propagation, debounce, throttle);
const Attribute$isEvent = (value)=>value instanceof Event;
const Attribute$Event$kind = (value)=>value.kind;
const Attribute$Event$0 = (value)=>value.kind;
const Attribute$Event$name = (value)=>value.name;
const Attribute$Event$1 = (value)=>value.name;
const Attribute$Event$handler = (value)=>value.handler;
const Attribute$Event$2 = (value)=>value.handler;
const Attribute$Event$include = (value)=>value.include;
const Attribute$Event$3 = (value)=>value.include;
const Attribute$Event$prevent_default = (value)=>value.prevent_default;
const Attribute$Event$4 = (value)=>value.prevent_default;
const Attribute$Event$stop_propagation = (value)=>value.stop_propagation;
const Attribute$Event$5 = (value)=>value.stop_propagation;
const Attribute$Event$debounce = (value)=>value.debounce;
const Attribute$Event$6 = (value)=>value.debounce;
const Attribute$Event$throttle = (value)=>value.throttle;
const Attribute$Event$7 = (value)=>value.throttle;
const Attribute$kind = (value)=>value.kind;
const Attribute$name = (value)=>value.name;
class Handler extends (0, _gleamMjs.CustomType) {
    constructor(prevent_default, stop_propagation, message){
        super();
        this.prevent_default = prevent_default;
        this.stop_propagation = stop_propagation;
        this.message = message;
    }
}
const Handler$Handler = (prevent_default, stop_propagation, message)=>new Handler(prevent_default, stop_propagation, message);
const Handler$isHandler = (value)=>value instanceof Handler;
const Handler$Handler$prevent_default = (value)=>value.prevent_default;
const Handler$Handler$0 = (value)=>value.prevent_default;
const Handler$Handler$stop_propagation = (value)=>value.stop_propagation;
const Handler$Handler$1 = (value)=>value.stop_propagation;
const Handler$Handler$message = (value)=>value.message;
const Handler$Handler$2 = (value)=>value.message;
class Never extends (0, _gleamMjs.CustomType) {
    constructor(kind){
        super();
        this.kind = kind;
    }
}
const EventBehaviour$Never = (kind)=>new Never(kind);
const EventBehaviour$isNever = (value)=>value instanceof Never;
const EventBehaviour$Never$kind = (value)=>value.kind;
const EventBehaviour$Never$0 = (value)=>value.kind;
class Possible extends (0, _gleamMjs.CustomType) {
    constructor(kind){
        super();
        this.kind = kind;
    }
}
const EventBehaviour$Possible = (kind)=>new Possible(kind);
const EventBehaviour$isPossible = (value)=>value instanceof Possible;
const EventBehaviour$Possible$kind = (value)=>value.kind;
const EventBehaviour$Possible$0 = (value)=>value.kind;
class Always extends (0, _gleamMjs.CustomType) {
    constructor(kind){
        super();
        this.kind = kind;
    }
}
const EventBehaviour$Always = (kind)=>new Always(kind);
const EventBehaviour$isAlways = (value)=>value instanceof Always;
const EventBehaviour$Always$kind = (value)=>value.kind;
const EventBehaviour$Always$0 = (value)=>value.kind;
const EventBehaviour$kind = (value)=>value.kind;
const attribute_kind = 0;
const property_kind = 1;
const event_kind = 2;
const never_kind = 0;
const never = /* @__PURE__ */ new Never(never_kind);
const possible_kind = 1;
const possible = /* @__PURE__ */ new Possible(possible_kind);
const always_kind = 2;
const always = /* @__PURE__ */ new Always(always_kind);
function merge(loop$attributes, loop$merged) {
    while(true){
        let attributes = loop$attributes;
        let merged = loop$merged;
        if (attributes instanceof (0, _gleamMjs.Empty)) return merged;
        else {
            let $ = attributes.head;
            if ($ instanceof Attribute) {
                let $1 = $.name;
                if ($1 === "") {
                    let rest = attributes.tail;
                    loop$attributes = rest;
                    loop$merged = merged;
                } else if ($1 === "class") {
                    let $2 = $.value;
                    if ($2 === "") {
                        let rest = attributes.tail;
                        loop$attributes = rest;
                        loop$merged = merged;
                    } else {
                        let $3 = attributes.tail;
                        if ($3 instanceof (0, _gleamMjs.Empty)) {
                            let attribute$1 = $;
                            let rest = $3;
                            loop$attributes = rest;
                            loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                        } else {
                            let $4 = $3.head;
                            if ($4 instanceof Attribute) {
                                let $5 = $4.name;
                                if ($5 === "class") {
                                    let kind = $.kind;
                                    let class1 = $2;
                                    let rest = $3.tail;
                                    let class2 = $4.value;
                                    let value = class1 + " " + class2;
                                    let attribute$1 = new Attribute(kind, "class", value);
                                    loop$attributes = (0, _gleamMjs.prepend)(attribute$1, rest);
                                    loop$merged = merged;
                                } else {
                                    let attribute$1 = $;
                                    let rest = $3;
                                    loop$attributes = rest;
                                    loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                                }
                            } else {
                                let attribute$1 = $;
                                let rest = $3;
                                loop$attributes = rest;
                                loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                            }
                        }
                    }
                } else if ($1 === "style") {
                    let $2 = $.value;
                    if ($2 === "") {
                        let rest = attributes.tail;
                        loop$attributes = rest;
                        loop$merged = merged;
                    } else {
                        let $3 = attributes.tail;
                        if ($3 instanceof (0, _gleamMjs.Empty)) {
                            let attribute$1 = $;
                            let rest = $3;
                            loop$attributes = rest;
                            loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                        } else {
                            let $4 = $3.head;
                            if ($4 instanceof Attribute) {
                                let $5 = $4.name;
                                if ($5 === "style") {
                                    let kind = $.kind;
                                    let style1 = $2;
                                    let rest = $3.tail;
                                    let style2 = $4.value;
                                    let value = style1 + ";" + style2;
                                    let attribute$1 = new Attribute(kind, "style", value);
                                    loop$attributes = (0, _gleamMjs.prepend)(attribute$1, rest);
                                    loop$merged = merged;
                                } else {
                                    let attribute$1 = $;
                                    let rest = $3;
                                    loop$attributes = rest;
                                    loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                                }
                            } else {
                                let attribute$1 = $;
                                let rest = $3;
                                loop$attributes = rest;
                                loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                            }
                        }
                    }
                } else {
                    let attribute$1 = $;
                    let rest = attributes.tail;
                    loop$attributes = rest;
                    loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
                }
            } else {
                let attribute$1 = $;
                let rest = attributes.tail;
                loop$attributes = rest;
                loop$merged = (0, _gleamMjs.prepend)(attribute$1, merged);
            }
        }
    }
}
function prepare(attributes) {
    if (attributes instanceof (0, _gleamMjs.Empty)) return attributes;
    else {
        let $ = attributes.tail;
        if ($ instanceof (0, _gleamMjs.Empty)) return attributes;
        else {
            let _pipe = attributes;
            let _pipe$1 = _listMjs.sort(_pipe, (a, b)=>{
                return (0, _vattrFfiMjs.compare)(b, a);
            });
            return merge(_pipe$1, _constantsMjs.empty_list);
        }
    }
}
function attribute_to_json(kind, name, value) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "name", name);
    let _pipe$2 = _jsonObjectBuilderMjs.string(_pipe$1, "value", value);
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function property_to_json(kind, name, value) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "name", name);
    let _pipe$2 = _jsonObjectBuilderMjs.json(_pipe$1, "value", value);
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function attribute(name, value) {
    return new Attribute(attribute_kind, name, value);
}
function to_string_tree(key, namespace, parent_namespace, attributes) {
    let _block;
    let $ = key !== "";
    if ($) _block = (0, _gleamMjs.prepend)(attribute("data-lustre-key", key), attributes);
    else _block = attributes;
    let attributes$1 = _block;
    let _block$1;
    let $1 = namespace !== parent_namespace;
    if ($1) {
        if (namespace === "") _block$1 = (0, _gleamMjs.prepend)(attribute("xmlns", "http://www.w3.org/1999/xhtml"), attributes$1);
        else _block$1 = (0, _gleamMjs.prepend)(attribute("xmlns", namespace), attributes$1);
    } else _block$1 = attributes$1;
    let attributes$2 = _block$1;
    return _listMjs.fold(attributes$2, _stringTreeMjs.new$(), (html, attr)=>{
        if (attr instanceof Attribute) {
            let $2 = attr.name;
            if ($2 === "virtual:defaultValue") {
                let value = attr.value;
                return _stringTreeMjs.append(html, " value=\"" + _houdiniMjs.escape(value) + "\"");
            } else if ($2 === "virtual:defaultChecked") return _stringTreeMjs.append(html, " checked");
            else if ($2 === "virtual:defaultSelected") return _stringTreeMjs.append(html, " selected");
            else if ($2 === "") return html;
            else {
                let $3 = attr.value;
                if ($3 === "") {
                    let name = $2;
                    return _stringTreeMjs.append(html, " " + name);
                } else {
                    let name = $2;
                    let value = $3;
                    return _stringTreeMjs.append(html, " " + name + "=\"" + _houdiniMjs.escape(value) + "\"");
                }
            }
        } else return html;
    });
}
function property(name, value) {
    return new Property(property_kind, name, value);
}
function event(name, handler, include, prevent_default, stop_propagation, debounce, throttle) {
    return new Event(event_kind, name, handler, include, prevent_default, stop_propagation, debounce, throttle);
}
function event_behaviour_to_json_builder(behaviour) {
    if (behaviour instanceof Never) {
        let kind = behaviour.kind;
        return _jsonObjectBuilderMjs.tagged(kind);
    } else if (behaviour instanceof Possible) return _jsonObjectBuilderMjs.tagged(never_kind);
    else {
        let kind = behaviour.kind;
        return _jsonObjectBuilderMjs.tagged(kind);
    }
}
function event_to_json(kind, name, include, prevent_default, stop_propagation, debounce, throttle) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "name", name);
    let _pipe$2 = _jsonObjectBuilderMjs.list(_pipe$1, "include", include, _jsonMjs.string);
    let _pipe$3 = _jsonObjectBuilderMjs.object(_pipe$2, "prevent_default", event_behaviour_to_json_builder(prevent_default));
    let _pipe$4 = _jsonObjectBuilderMjs.object(_pipe$3, "stop_propagation", event_behaviour_to_json_builder(stop_propagation));
    let _pipe$5 = _jsonObjectBuilderMjs.int(_pipe$4, "debounce", debounce);
    let _pipe$6 = _jsonObjectBuilderMjs.int(_pipe$5, "throttle", throttle);
    return _jsonObjectBuilderMjs.build(_pipe$6);
}
function to_json(attribute) {
    if (attribute instanceof Attribute) {
        let kind = attribute.kind;
        let name = attribute.name;
        let value = attribute.value;
        return attribute_to_json(kind, name, value);
    } else if (attribute instanceof Property) {
        let kind = attribute.kind;
        let name = attribute.name;
        let value = attribute.value;
        return property_to_json(kind, name, value);
    } else {
        let kind = attribute.kind;
        let name = attribute.name;
        let include = attribute.include;
        let prevent_default = attribute.prevent_default;
        let stop_propagation = attribute.stop_propagation;
        let debounce = attribute.debounce;
        let throttle = attribute.throttle;
        return event_to_json(kind, name, include, prevent_default, stop_propagation, debounce, throttle);
    }
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../../gleam_stdlib/gleam/order.mjs":"eYj92","../../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../../gleam_stdlib/gleam/string_tree.mjs":"8IH0o","../../../houdini/houdini.mjs":"e94ou","../../gleam.mjs":"jNPQG","../../lustre/internals/constants.mjs":"gKFR6","../../lustre/internals/json_object_builder.mjs":"31ZqD","./vattr.ffi.mjs":"9YJNf","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"e94ou":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Escapes a string to be safely used inside an HTML document by escaping
 * the following characters:
 *   - `<` becomes `&lt;`
 *   - `>` becomes `&gt;`
 *   - `&` becomes `&amp;`
 *   - `"` becomes `&quot;`
 *   - `'` becomes `&#39;`.
 *
 * ## Examples
 *
 * ```gleam
 * assert escape("wibble & wobble") == "wibble &amp; wobble"
 * assert escape("wibble > wobble") == "wibble &gt; wobble"
 * ```
 */ parcelHelpers.export(exports, "escape", ()=>escape);
var _escapeJsMjs = require("./houdini/internal/escape_js.mjs");
function escape(string) {
    return _escapeJsMjs.escape(string);
}

},{"./houdini/internal/escape_js.mjs":"5xy9L","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5xy9L":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * This `escape` function will work on all targets, beware that the version
 * specifically optimised for Erlang will be _way faster_ than this one when
 * running on the BEAM. That's why this fallback implementation is only ever
 * used when running on the JS backend.
 */ parcelHelpers.export(exports, "escape", ()=>escape);
var _houdiniFfiMjs = require("../../houdini.ffi.mjs");
function escape(text) {
    return (0, _houdiniFfiMjs.do_escape)(text);
}

},{"../../houdini.ffi.mjs":"jRCTb","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"jRCTb":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "do_escape", ()=>do_escape);
function do_escape(string) {
    return string.replaceAll(/[><&"']/g, (replaced)=>{
        switch(replaced){
            case ">":
                return "&gt;";
            case "<":
                return "&lt;";
            case "'":
                return "&#39;";
            case "&":
                return "&amp;";
            case '"':
                return "&quot;";
            default:
                return replaced;
        }
    });
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"gKFR6":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "empty_list", ()=>empty_list);
parcelHelpers.export(exports, "error_nil", ()=>error_nil);
var _gleamMjs = require("../../gleam.mjs");
const empty_list = /* @__PURE__ */ (0, _gleamMjs.toList)([]);
const error_nil = /* @__PURE__ */ new (0, _gleamMjs.Error)(undefined);

},{"../../gleam.mjs":"jNPQG","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"31ZqD":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "new$", ()=>new$);
parcelHelpers.export(exports, "json", ()=>json);
parcelHelpers.export(exports, "tagged", ()=>tagged);
parcelHelpers.export(exports, "build", ()=>build);
parcelHelpers.export(exports, "string", ()=>string);
parcelHelpers.export(exports, "int", ()=>int);
parcelHelpers.export(exports, "bool", ()=>bool);
parcelHelpers.export(exports, "list", ()=>list);
parcelHelpers.export(exports, "object", ()=>object);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
function new$() {
    return _constantsMjs.empty_list;
}
function json(entries, key, value) {
    return (0, _gleamMjs.prepend)([
        key,
        value
    ], entries);
}
function tagged(kind) {
    return (0, _gleamMjs.toList)([
        [
            "kind",
            _jsonMjs.int(kind)
        ]
    ]);
}
function build(entries) {
    return _jsonMjs.object(entries);
}
function string(entries, key, value) {
    let $ = value !== "";
    if ($) return (0, _gleamMjs.prepend)([
        key,
        _jsonMjs.string(value)
    ], entries);
    else return entries;
}
function int(entries, key, value) {
    let $ = value !== 0;
    if ($) return (0, _gleamMjs.prepend)([
        key,
        _jsonMjs.int(value)
    ], entries);
    else return entries;
}
function bool(entries, key, value) {
    if (value) return (0, _gleamMjs.prepend)([
        key,
        _jsonMjs.int(1)
    ], entries);
    else return entries;
}
function list(entries, key, values, to_json) {
    if (values instanceof (0, _gleamMjs.Empty)) return entries;
    else return (0, _gleamMjs.prepend)([
        key,
        _jsonMjs.array(values, to_json)
    ], entries);
}
function object(entries, key, nested) {
    if (nested instanceof (0, _gleamMjs.Empty)) return entries;
    else return (0, _gleamMjs.prepend)([
        key,
        _jsonMjs.object(nested)
    ], entries);
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam.mjs":"jNPQG","../../lustre/internals/constants.mjs":"gKFR6","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"9YJNf":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "compare", ()=>compare);
var _orderMjs = require("../../../gleam_stdlib/gleam/order.mjs");
const GT = /* @__PURE__ */ (0, _orderMjs.Order$Gt)();
const LT = /* @__PURE__ */ (0, _orderMjs.Order$Lt)();
const EQ = /* @__PURE__ */ (0, _orderMjs.Order$Eq)();
function compare(a, b) {
    if (a.name === b.name) return EQ;
    else if (a.name < b.name) return LT;
    else return GT;
}

},{"../../../gleam_stdlib/gleam/order.mjs":"eYj92","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"iAEPi":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Transform the result of an effect. This is useful for mapping over effects
 * produced by other libraries or modules.
 *
 * > **Note**: Remember that effects are not _required_ to dispatch any messages.
 * > Your mapping function may never be called!
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Perform a side effect by supplying your own `dispatch` and `emit`functions.
 * This is primarily used internally by the server component runtime, but it is
 * may also useful for testing.
 *
 * Because this is run outside of the runtime, timing-related effects scheduled
 * by `before_paint` and `after_paint` will **not** be run.
 *
 * > **Note**: For now, you should **not** consider this function a part of the
 * > public API. It may be removed in a future minor or patch release. If you have
 * > a specific use case for this function, we'd love to hear about it! Please
 * > reach out on the [Gleam Discord](https://discord.gg/Fm8Pwmy) or
 * > [open an issue](https://github.com/lustre-labs/lustre/issues/new)!
 * 
 * @ignore
 */ parcelHelpers.export(exports, "perform", ()=>perform);
/**
 * Most Lustre applications need to return a tuple of `#(model, Effect(msg))`
 * from their `init` and `update` functions. If you don't want to perform any
 * side effects, you can use `none` to tell the runtime there's no work to do.
 */ parcelHelpers.export(exports, "none", ()=>none);
/**
 * Construct your own reusable effect from a custom callback. This callback is
 * called with a `dispatch` function you can use to send messages back to your
 * application's `update` function.
 *
 * Example using the `window` module from the `plinth` library to dispatch a
 * message on the browser window object's `"visibilitychange"` event.
 *
 * ```gleam
 * import lustre/effect.{type Effect}
 * import plinth/browser/window
 *
 * type Model {
 *   Model(Int)
 * }
 *
 * type Msg {
 *   FetchState
 * }
 *
 * fn init(_flags) -> #(Model, Effect(Msg)) {
 *   #(
 *     Model(0),
 *     effect.from(fn(dispatch) {
 *       window.add_event_listener("visibilitychange", fn(_event) {
 *         dispatch(FetchState)
 *       })
 *     }),
 *   )
 * }
 * ```
 */ parcelHelpers.export(exports, "from", ()=>from);
/**
 * Schedule a side effect that is guaranteed to run after your `view` function
 * is called and the DOM has been updated, but **before** the browser has
 * painted the screen. This effect is useful when you need to read from the DOM
 * or perform other operations that might affect the layout of your application.
 *
 * In addition to the `dispatch` function, your callback will also be provided
 * with root element of your app or component. This is especially useful inside
 * of components, giving you a reference to the [Shadow Root](https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot).
 *
 * Messages dispatched immediately in this effect will trigger a second re-render
 * of your application before the browser paints the screen. This let's you read
 * the state of the DOM, update your model, and then render a second time with
 * the additional information.
 *
 * > **Note**: dispatching messages synchronously in this effect can lead to
 * > degraded performance if not used correctly. In the worst case you can lock
 * > up the browser and prevent it from painting the screen _at all_.
 *
 * > **Note**: There is no concept of a "paint" for server components. These
 * > effects will be ignored in those contexts and never run.
 */ parcelHelpers.export(exports, "before_paint", ()=>before_paint);
/**
 * Schedule a side effect that is guaranteed to run after the browser has painted
 * the screen.
 *
 * In addition to the `dispatch` function, your callback will also be provided
 * with root element of your app or component. This is especially useful inside
 * of components, giving you a reference to the [Shadow Root](https://developer.mozilla.org/en-US/docs/Web/API/ShadowRoot).
 *
 * > **Note**: There is no concept of a "paint" for server components. These
 * > effects will be ignored in those contexts and never run.
 */ parcelHelpers.export(exports, "after_paint", ()=>after_paint);
/**
 * Emit a custom event from a component as an effect. Parents can listen to these
 * events in their `view` function like any other HTML event. Any data you pass
 * to `effect.emit` can be accessed by event listeners through the `detail` property
 * of the event object.
 * 
 * @ignore
 */ parcelHelpers.export(exports, "event", ()=>event);
parcelHelpers.export(exports, "select", ()=>select);
parcelHelpers.export(exports, "provide", ()=>provide);
/**
 * Batch multiple effects to be performed at the same time.
 *
 * > **Note**: The runtime makes no guarantees about the order on which effects
 * > are performed! If you need to chain or sequence effects together, you have
 * > two broad options:
 * >
 * > 1. Create variants of your `msg` type to represent each step in the sequence
 * >    and fire off the next effect in response to the previous one.
 * >
 * > 2. If you're defining effects yourself, consider whether or not you can handle
 * >    the sequencing inside the effect itself.
 */ parcelHelpers.export(exports, "batch", ()=>batch);
var _processMjs = require("../../gleam_erlang/gleam/erlang/process.mjs");
var _jsonMjs = require("../../gleam_json/gleam/json.mjs");
var _dynamicMjs = require("../../gleam_stdlib/gleam/dynamic.mjs");
var _listMjs = require("../../gleam_stdlib/gleam/list.mjs");
var _gleamMjs = require("../gleam.mjs");
class Effect extends (0, _gleamMjs.CustomType) {
    constructor(synchronous, before_paint, after_paint){
        super();
        this.synchronous = synchronous;
        this.before_paint = before_paint;
        this.after_paint = after_paint;
    }
}
class Actions extends (0, _gleamMjs.CustomType) {
    constructor(dispatch, emit, select, root, provide){
        super();
        this.dispatch = dispatch;
        this.emit = emit;
        this.select = select;
        this.root = root;
        this.provide = provide;
    }
}
const empty = /* @__PURE__ */ new Effect(/* @__PURE__ */ (0, _gleamMjs.toList)([]), /* @__PURE__ */ (0, _gleamMjs.toList)([]), /* @__PURE__ */ (0, _gleamMjs.toList)([]));
function do_comap_select(_, _1, _2) {
    return undefined;
}
function do_comap_actions(actions, f) {
    return new Actions((msg)=>{
        return actions.dispatch(f(msg));
    }, actions.emit, (selector)=>{
        return do_comap_select(actions, selector, f);
    }, actions.root, actions.provide);
}
function do_map(effects, f) {
    return _listMjs.map(effects, (effect)=>{
        return (actions)=>{
            return effect(do_comap_actions(actions, f));
        };
    });
}
function map(effect, f) {
    return new Effect(do_map(effect.synchronous, f), do_map(effect.before_paint, f), do_map(effect.after_paint, f));
}
function perform(effect, dispatch, emit, select, root, provide) {
    let actions = new Actions(dispatch, emit, select, root, provide);
    return _listMjs.each(effect.synchronous, (run)=>{
        return run(actions);
    });
}
function none() {
    return empty;
}
function from(effect) {
    let task = (actions)=>{
        let dispatch = actions.dispatch;
        return effect(dispatch);
    };
    return new Effect((0, _gleamMjs.toList)([
        task
    ]), empty.before_paint, empty.after_paint);
}
function before_paint(effect) {
    let task = (actions)=>{
        let root = actions.root();
        let dispatch = actions.dispatch;
        return effect(dispatch, root);
    };
    return new Effect(empty.synchronous, (0, _gleamMjs.toList)([
        task
    ]), empty.after_paint);
}
function after_paint(effect) {
    let task = (actions)=>{
        let root = actions.root();
        let dispatch = actions.dispatch;
        return effect(dispatch, root);
    };
    return new Effect(empty.synchronous, empty.before_paint, (0, _gleamMjs.toList)([
        task
    ]));
}
function event(name, data) {
    let task = (actions)=>{
        return actions.emit(name, data);
    };
    return new Effect((0, _gleamMjs.toList)([
        task
    ]), empty.before_paint, empty.after_paint);
}
function select(_) {
    return empty;
}
function provide(key, value) {
    let task = (actions)=>{
        return actions.provide(key, value);
    };
    return new Effect((0, _gleamMjs.toList)([
        task
    ]), empty.before_paint, empty.after_paint);
}
function batch(effects) {
    return _listMjs.fold(effects, empty, (acc, eff)=>{
        return new Effect(_listMjs.fold(eff.synchronous, acc.synchronous, _listMjs.prepend), _listMjs.fold(eff.before_paint, acc.before_paint, _listMjs.prepend), _listMjs.fold(eff.after_paint, acc.after_paint, _listMjs.prepend));
    });
}

},{"../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../gleam_stdlib/gleam/list.mjs":"8dUwY","../gleam.mjs":"jNPQG","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"2XxJ4":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * A general function for constructing any kind of element. In most cases you
 * will want to use the [`lustre/element/html`](./element/html.html) instead but this
 * function is particularly handy when constructing custom elements, either
 * from your own Lustre components or from external JavaScript libraries.
 *
 * > **Note**: Because Lustre is primarily used to create HTML, this function
 * > special-cases the following tags which render as
 * > [void elements](https://developer.mozilla.org/en-US/docs/Glossary/Void_element):
 * >
 * >   - area
 * >   - base
 * >   - br
 * >   - col
 * >   - embed
 * >   - hr
 * >   - img
 * >   - input
 * >   - link
 * >   - meta
 * >   - param
 * >   - source
 * >   - track
 * >   - wbr
 * >
 * > This will only affect the output of `to_string` and `to_string_builder`!
 * > If you need to render any of these tags with children, *or* you want to
 * > render some other tag as self-closing or void, use [`advanced`](#advanced)
 * > to construct the element instead.
 */ parcelHelpers.export(exports, "element", ()=>element);
/**
 * A function for constructing elements in a specific XML namespace. This can
 * be used to construct SVG or MathML elements, for example.
 */ parcelHelpers.export(exports, "namespaced", ()=>namespaced);
/**
 * A function for constructing elements with more control over how the element
 * is rendered when converted to a string. This is necessary because some HTML,
 * SVG, and MathML elements are self-closing or void elements, and Lustre needs
 * to know how to render them correctly!
 */ parcelHelpers.export(exports, "advanced", ()=>advanced);
/**
 * A function for turning a Gleam string into a text node. Gleam doesn't have
 * union types like some other languages you may be familiar with, like TypeScript.
 * Instead, we need a way to take a `String` and turn it into an `Element` somehow:
 * this function is exactly that!
 */ parcelHelpers.export(exports, "text", ()=>text);
/**
 * A function for rendering nothing. This is mostly useful for conditional
 * rendering, where you might want to render something only if a certain
 * condition is met.
 */ parcelHelpers.export(exports, "none", ()=>none);
/**
 * A function for constructing a wrapper element with no tag name. This is
 * useful for wrapping a list of elements together without adding an extra
 * `<div>` or other container element, or returning multiple elements in places
 * where only one `Element` is expected.
 */ parcelHelpers.export(exports, "fragment", ()=>fragment);
/**
 * A function for constructing a wrapper element with custom raw HTML as its
 * content. Lustre will render the provided HTML verbatim, and will not touch
 * its children except when replacing the entire inner html on changes.
 *
 * For HTML elements you can use an empty string for the namespace.
 *
 * > **Note:** The provided HTML will not be escaped automatically and may expose
 * > your applications to XSS attacks! Make sure you absolutely trust the HTML you
 * > pass to this function. In particular, never use this to display un-sanitised
 * > user HTML!
 */ parcelHelpers.export(exports, "unsafe_raw_html", ()=>unsafe_raw_html);
/**
 * A function for creating "memoised" or "lazy" elements. Lustre will use the
 * dependencies list to skip calling the provided view function if all of the
 * dependencies a _reference equal_ to their previous values.
 *
 * `memo` can be used to optimise performance-critical parts of your application,
 * for example in cases where many instances of the same element are rendered but
 * only one may change at a time, or cases where a part of your view may update
 * very frequently but other parts remain largely static. When Lustre can tell
 * that the dependencies haven't changed, almost all the work typically done to
 * update the DOM can be skipped.
 *
 * In many cases `memo` will not be necessary, so think twice before considering
 * its use! Lustre is designed to handle rerenders and large vdom trees efficiently,
 * so in most cases the naive approach of re-rendering everything will be perfectly
 * fine.
 *
 * > **Note**: reference equality is not the same as Gleam's normal equality.
 * > Two custom types with the same values are not reference equal unless they
 * > are the exact same instance in memory! Because of this, it's important to
 * > avoid list literals or constructing custom types in the dependencies list.
 *
 * > **Note**: memoisation comes with its own trade-offs and can cause performance
 * > regressions in two ways. First, every use of `memo` increases your application's
 * > memory usage slightly, as Lustre needs to keep dependencies around to compare
 * > them on subsequent renders. Second, if dependencies change regularly, the
 * > overhead of comparing dependencies and managing memoisation may be more than
 * > the naive cost of re-rendering the element each time.
 */ parcelHelpers.export(exports, "memo", ()=>memo);
/**
 * Create a `Ref` dependency value used for [`memo`](#memo) elements.
 *
 * Lustre uses reference equality to compare dependencies. On JavaScript, values
 * are compared using [same-value-zero](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Equality_comparisons_and_sameness#same-value-zero_equality)
 * semantics. This means Lustre will treat `+0` and `-0` as equal, and any errant
 * `NaN` values (which are not typically producible in Gleam code) as equal. On
 * Erlang, there is no difference between reference equality and value equality,
 * so all values are compared using normal equality semantics.
 */ parcelHelpers.export(exports, "ref", ()=>ref);
/**
 * The `Element` type is parameterised by the type of messages it can produce
 * from events. Sometimes you might end up with a fragment of HTML from another
 * library or module that produces a different type of message: this function lets
 * you map the messages produced from one type to another.
 *
 * Think of it like `list.map` or `result.map` but for HTML events!
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Convert a Lustre `Element` to a string. This is _not_ pretty-printed, so
 * there are no newlines or indentation. If you need to pretty-print an element,
 * reach out on the [Gleam Discord](https://discord.gg/Fm8Pwmy) or
 * [open an issue](https://github.com/lustre-labs/lustre/issues/new) with your
 * use case and we'll see what we can do!
 */ parcelHelpers.export(exports, "to_string", ()=>to_string);
/**
 * Converts an element to a string like [`to_string`](#to_string), but prepends
 * a `<!doctype html>` declaration to the string. This is useful for rendering
 * complete HTML documents.
 *
 * If the provided element is not an `html` element, it will be wrapped in both
 * a `html` and `body` element.
 */ parcelHelpers.export(exports, "to_document_string", ()=>to_document_string);
/**
 * Convert a Lustre `Element` to a `StringTree`. This is _not_ pretty-printed,
 * so there are no newlines or indentation. If you need to pretty-print an element,
 * reach out on the [Gleam Discord](https://discord.gg/Fm8Pwmy) or
 * [open an issue](https://github.com/lustre-labs/lustre/issues/new) with your
 * use case and we'll see what we can do!
 */ parcelHelpers.export(exports, "to_string_tree", ()=>to_string_tree);
/**
 * Converts an element to a `StringTree` like [`to_string_builder`](#to_string_builder),
 * but prepends a `<!doctype html>` declaration. This is useful for rendering
 * complete HTML documents.
 *
 * If the provided element is not an `html` element, it will be wrapped in both
 * a `html` and `body` element.
 */ parcelHelpers.export(exports, "to_document_string_tree", ()=>to_document_string_tree);
/**
 * Converts a Lustre `Element` to a human-readable string by inserting new lines
 * and indentation where appropriate. This is useful for debugging and testing,
 * but for production code you should use [`to_string`](#to_string) or
 * [`to_document_string`](#to_document_string) instead.
 *
 * 💡 This function works great with the snapshot testing library
 *    [birdie](https://hexdocs.pm/birdie)!
 *
 * ## Using `to_string`:
 *
 * ```html
 * <header><h1>Hello, world!</h1></header>
 * ```
 *
 * ## Using `to_readable_string`
 *
 * ```html
 * <header>
 *   <h1>
 *     Hello, world!
 *   </h1>
 * </header>
 * ```
 */ parcelHelpers.export(exports, "to_readable_string", ()=>to_readable_string);
var _stringMjs = require("../../gleam_stdlib/gleam/string.mjs");
var _stringTreeMjs = require("../../gleam_stdlib/gleam/string_tree.mjs");
var _gleamMjs = require("../gleam.mjs");
var _attributeMjs = require("../lustre/attribute.mjs");
var _mutableMapMjs = require("../lustre/internals/mutable_map.mjs");
var _refMjs = require("../lustre/internals/ref.mjs");
var _vnodeMjs = require("../lustre/vdom/vnode.mjs");
function element(tag, attributes, children) {
    return _vnodeMjs.element("", "", tag, attributes, children, _mutableMapMjs.new$(), false, _vnodeMjs.is_void_html_element(tag, ""));
}
function namespaced(namespace, tag, attributes, children) {
    return _vnodeMjs.element("", namespace, tag, attributes, children, _mutableMapMjs.new$(), false, _vnodeMjs.is_void_html_element(tag, namespace));
}
function advanced(namespace, tag, attributes, children, self_closing, void$) {
    return _vnodeMjs.element("", namespace, tag, attributes, children, _mutableMapMjs.new$(), self_closing, void$);
}
function text(content) {
    return _vnodeMjs.text("", content);
}
function none() {
    return _vnodeMjs.text("", "");
}
function fragment(children) {
    return _vnodeMjs.fragment("", children, _mutableMapMjs.new$());
}
function unsafe_raw_html(namespace, tag, attributes, inner_html) {
    return _vnodeMjs.unsafe_inner_html("", namespace, tag, attributes, inner_html);
}
function memo(dependencies, view) {
    return _vnodeMjs.memo("", dependencies, view);
}
function ref(value) {
    return _refMjs.from(value);
}
function map(element, f) {
    return _vnodeMjs.map(element, f);
}
function to_string(element) {
    return _vnodeMjs.to_string(element);
}
function to_document_string(el) {
    let _pipe = _vnodeMjs.to_string((()=>{
        if (el instanceof (0, _vnodeMjs.Element)) {
            let $ = el.tag;
            if ($ === "html") return el;
            else if ($ === "head") return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]));
            else if ($ === "body") return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]));
            else return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                element("body", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                    el
                ]))
            ]));
        } else return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
            element("body", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]))
        ]));
    })());
    return ((_capture)=>{
        return _stringMjs.append("<!doctype html>\n", _capture);
    })(_pipe);
}
function to_string_tree(element) {
    return _vnodeMjs.to_string_tree(element, "");
}
function to_document_string_tree(el) {
    let _pipe = _vnodeMjs.to_string_tree((()=>{
        if (el instanceof (0, _vnodeMjs.Element)) {
            let $ = el.tag;
            if ($ === "html") return el;
            else if ($ === "head") return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]));
            else if ($ === "body") return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]));
            else return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                element("body", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                    el
                ]))
            ]));
        } else return element("html", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
            element("body", (0, _gleamMjs.toList)([]), (0, _gleamMjs.toList)([
                el
            ]))
        ]));
    })(), "");
    return _stringTreeMjs.prepend(_pipe, "<!doctype html>\n");
}
function to_readable_string(el) {
    return _vnodeMjs.to_snapshot(el, false);
}

},{"../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../gleam_stdlib/gleam/string_tree.mjs":"8IH0o","../gleam.mjs":"jNPQG","../lustre/attribute.mjs":"faRXj","../lustre/internals/mutable_map.mjs":"6NvMa","../lustre/internals/ref.mjs":"gnct2","../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"6NvMa":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "delete$", ()=>(0, _mutableMapFfiMjs.remove));
parcelHelpers.export(exports, "get_or_compute", ()=>(0, _mutableMapFfiMjs.get_or_compute));
parcelHelpers.export(exports, "has_key", ()=>(0, _mutableMapFfiMjs.has_key));
parcelHelpers.export(exports, "insert", ()=>(0, _mutableMapFfiMjs.insert));
parcelHelpers.export(exports, "new$", ()=>(0, _mutableMapFfiMjs.empty));
parcelHelpers.export(exports, "size", ()=>(0, _mutableMapFfiMjs.size));
parcelHelpers.export(exports, "unsafe_get", ()=>(0, _mutableMapFfiMjs.get));
/**
 *
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
var _mutableMapFfiMjs = require("./mutable_map.ffi.mjs");
function is_empty(map) {
    return (0, _mutableMapFfiMjs.size)(map) === 0;
}

},{"./mutable_map.ffi.mjs":"eatSe","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eatSe":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "empty", ()=>empty);
parcelHelpers.export(exports, "get", ()=>get);
parcelHelpers.export(exports, "get_or_compute", ()=>get_or_compute);
parcelHelpers.export(exports, "has_key", ()=>has_key);
parcelHelpers.export(exports, "insert", ()=>insert);
parcelHelpers.export(exports, "remove", ()=>remove);
parcelHelpers.export(exports, "size", ()=>size);
function empty() {
    return null;
}
function get(map, key) {
    return map?.get(key);
}
function get_or_compute(map, key, compute) {
    return map?.get(key) ?? compute();
}
function has_key(map, key) {
    return map && map.has(key);
}
function insert(map, key, value) {
    map ??= new Map();
    map.set(key, value);
    return map;
}
function remove(map, key) {
    map?.delete(key);
    return map;
}
function size(map) {
    return map ? map.size : 0;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"gnct2":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "equal", ()=>(0, _refFfiMjs.sameValueZero));
parcelHelpers.export(exports, "from", ()=>(0, _functionMjs.identity));
/**
 *
 */ parcelHelpers.export(exports, "equal_lists", ()=>equal_lists);
var _functionMjs = require("../../../gleam_stdlib/gleam/function.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _refFfiMjs = require("./ref.ffi.mjs");
function equal_lists(loop$xs, loop$ys) {
    while(true){
        let xs = loop$xs;
        let ys = loop$ys;
        if (xs instanceof (0, _gleamMjs.Empty)) {
            if (ys instanceof (0, _gleamMjs.Empty)) return true;
            else return false;
        } else if (ys instanceof (0, _gleamMjs.Empty)) return false;
        else {
            let x = xs.head;
            let xs$1 = xs.tail;
            let y = ys.head;
            let ys$1 = ys.tail;
            let $ = (0, _refFfiMjs.sameValueZero)(x, y);
            if ($) {
                loop$xs = xs$1;
                loop$ys = ys$1;
            } else return $;
        }
    }
}

},{"../../../gleam_stdlib/gleam/function.mjs":"2jh6y","../../gleam.mjs":"jNPQG","./ref.ffi.mjs":"cQ9qf","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"cQ9qf":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "sameValueZero", ()=>sameValueZero);
function sameValueZero(x, y) {
    if (typeof x === "number" && typeof y === "number") // x and y are equal (may be -0 and 0) or they are both NaN
    return x === y || x !== x && y !== y;
    return x === y;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"j2vnp":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Fragment", ()=>Fragment);
parcelHelpers.export(exports, "Element$Fragment", ()=>Element$Fragment);
parcelHelpers.export(exports, "Element$isFragment", ()=>Element$isFragment);
parcelHelpers.export(exports, "Element$Fragment$kind", ()=>Element$Fragment$kind);
parcelHelpers.export(exports, "Element$Fragment$0", ()=>Element$Fragment$0);
parcelHelpers.export(exports, "Element$Fragment$key", ()=>Element$Fragment$key);
parcelHelpers.export(exports, "Element$Fragment$1", ()=>Element$Fragment$1);
parcelHelpers.export(exports, "Element$Fragment$children", ()=>Element$Fragment$children);
parcelHelpers.export(exports, "Element$Fragment$2", ()=>Element$Fragment$2);
parcelHelpers.export(exports, "Element$Fragment$keyed_children", ()=>Element$Fragment$keyed_children);
parcelHelpers.export(exports, "Element$Fragment$3", ()=>Element$Fragment$3);
parcelHelpers.export(exports, "Element", ()=>Element);
parcelHelpers.export(exports, "Element$Element", ()=>Element$Element);
parcelHelpers.export(exports, "Element$isElement", ()=>Element$isElement);
parcelHelpers.export(exports, "Element$Element$kind", ()=>Element$Element$kind);
parcelHelpers.export(exports, "Element$Element$0", ()=>Element$Element$0);
parcelHelpers.export(exports, "Element$Element$key", ()=>Element$Element$key);
parcelHelpers.export(exports, "Element$Element$1", ()=>Element$Element$1);
parcelHelpers.export(exports, "Element$Element$namespace", ()=>Element$Element$namespace);
parcelHelpers.export(exports, "Element$Element$2", ()=>Element$Element$2);
parcelHelpers.export(exports, "Element$Element$tag", ()=>Element$Element$tag);
parcelHelpers.export(exports, "Element$Element$3", ()=>Element$Element$3);
parcelHelpers.export(exports, "Element$Element$attributes", ()=>Element$Element$attributes);
parcelHelpers.export(exports, "Element$Element$4", ()=>Element$Element$4);
parcelHelpers.export(exports, "Element$Element$children", ()=>Element$Element$children);
parcelHelpers.export(exports, "Element$Element$5", ()=>Element$Element$5);
parcelHelpers.export(exports, "Element$Element$keyed_children", ()=>Element$Element$keyed_children);
parcelHelpers.export(exports, "Element$Element$6", ()=>Element$Element$6);
parcelHelpers.export(exports, "Element$Element$self_closing", ()=>Element$Element$self_closing);
parcelHelpers.export(exports, "Element$Element$7", ()=>Element$Element$7);
parcelHelpers.export(exports, "Element$Element$void", ()=>Element$Element$void);
parcelHelpers.export(exports, "Element$Element$8", ()=>Element$Element$8);
parcelHelpers.export(exports, "Text", ()=>Text);
parcelHelpers.export(exports, "Element$Text", ()=>Element$Text);
parcelHelpers.export(exports, "Element$isText", ()=>Element$isText);
parcelHelpers.export(exports, "Element$Text$kind", ()=>Element$Text$kind);
parcelHelpers.export(exports, "Element$Text$0", ()=>Element$Text$0);
parcelHelpers.export(exports, "Element$Text$key", ()=>Element$Text$key);
parcelHelpers.export(exports, "Element$Text$1", ()=>Element$Text$1);
parcelHelpers.export(exports, "Element$Text$content", ()=>Element$Text$content);
parcelHelpers.export(exports, "Element$Text$2", ()=>Element$Text$2);
parcelHelpers.export(exports, "UnsafeInnerHtml", ()=>UnsafeInnerHtml);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml", ()=>Element$UnsafeInnerHtml);
parcelHelpers.export(exports, "Element$isUnsafeInnerHtml", ()=>Element$isUnsafeInnerHtml);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$kind", ()=>Element$UnsafeInnerHtml$kind);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$0", ()=>Element$UnsafeInnerHtml$0);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$key", ()=>Element$UnsafeInnerHtml$key);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$1", ()=>Element$UnsafeInnerHtml$1);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$namespace", ()=>Element$UnsafeInnerHtml$namespace);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$2", ()=>Element$UnsafeInnerHtml$2);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$tag", ()=>Element$UnsafeInnerHtml$tag);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$3", ()=>Element$UnsafeInnerHtml$3);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$attributes", ()=>Element$UnsafeInnerHtml$attributes);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$4", ()=>Element$UnsafeInnerHtml$4);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$inner_html", ()=>Element$UnsafeInnerHtml$inner_html);
parcelHelpers.export(exports, "Element$UnsafeInnerHtml$5", ()=>Element$UnsafeInnerHtml$5);
parcelHelpers.export(exports, "Map", ()=>Map);
parcelHelpers.export(exports, "Element$Map", ()=>Element$Map);
parcelHelpers.export(exports, "Element$isMap", ()=>Element$isMap);
parcelHelpers.export(exports, "Element$Map$kind", ()=>Element$Map$kind);
parcelHelpers.export(exports, "Element$Map$0", ()=>Element$Map$0);
parcelHelpers.export(exports, "Element$Map$key", ()=>Element$Map$key);
parcelHelpers.export(exports, "Element$Map$1", ()=>Element$Map$1);
parcelHelpers.export(exports, "Element$Map$mapper", ()=>Element$Map$mapper);
parcelHelpers.export(exports, "Element$Map$2", ()=>Element$Map$2);
parcelHelpers.export(exports, "Element$Map$child", ()=>Element$Map$child);
parcelHelpers.export(exports, "Element$Map$3", ()=>Element$Map$3);
parcelHelpers.export(exports, "Memo", ()=>Memo);
parcelHelpers.export(exports, "Element$Memo", ()=>Element$Memo);
parcelHelpers.export(exports, "Element$isMemo", ()=>Element$isMemo);
parcelHelpers.export(exports, "Element$Memo$kind", ()=>Element$Memo$kind);
parcelHelpers.export(exports, "Element$Memo$0", ()=>Element$Memo$0);
parcelHelpers.export(exports, "Element$Memo$key", ()=>Element$Memo$key);
parcelHelpers.export(exports, "Element$Memo$1", ()=>Element$Memo$1);
parcelHelpers.export(exports, "Element$Memo$dependencies", ()=>Element$Memo$dependencies);
parcelHelpers.export(exports, "Element$Memo$2", ()=>Element$Memo$2);
parcelHelpers.export(exports, "Element$Memo$view", ()=>Element$Memo$view);
parcelHelpers.export(exports, "Element$Memo$3", ()=>Element$Memo$3);
parcelHelpers.export(exports, "Element$key", ()=>Element$key);
parcelHelpers.export(exports, "Element$kind", ()=>Element$kind);
parcelHelpers.export(exports, "fragment_kind", ()=>fragment_kind);
parcelHelpers.export(exports, "element_kind", ()=>element_kind);
parcelHelpers.export(exports, "text_kind", ()=>text_kind);
parcelHelpers.export(exports, "unsafe_inner_html_kind", ()=>unsafe_inner_html_kind);
parcelHelpers.export(exports, "map_kind", ()=>map_kind);
parcelHelpers.export(exports, "memo_kind", ()=>memo_kind);
parcelHelpers.export(exports, "is_void_html_element", ()=>is_void_html_element);
parcelHelpers.export(exports, "to_keyed", ()=>to_keyed);
parcelHelpers.export(exports, "fragment", ()=>fragment);
parcelHelpers.export(exports, "element", ()=>element);
parcelHelpers.export(exports, "text", ()=>text);
parcelHelpers.export(exports, "unsafe_inner_html", ()=>unsafe_inner_html);
parcelHelpers.export(exports, "map", ()=>map);
parcelHelpers.export(exports, "memo", ()=>memo);
parcelHelpers.export(exports, "to_snapshot", ()=>to_snapshot);
parcelHelpers.export(exports, "to_string_tree", ()=>to_string_tree);
parcelHelpers.export(exports, "to_string", ()=>to_string);
parcelHelpers.export(exports, "to_json", ()=>to_json);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _functionMjs = require("../../../gleam_stdlib/gleam/function.mjs");
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
var _stringMjs = require("../../../gleam_stdlib/gleam/string.mjs");
var _stringTreeMjs = require("../../../gleam_stdlib/gleam/string_tree.mjs");
var _houdiniMjs = require("../../../houdini/houdini.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _jsonObjectBuilderMjs = require("../../lustre/internals/json_object_builder.mjs");
var _mutableMapMjs = require("../../lustre/internals/mutable_map.mjs");
var _refMjs = require("../../lustre/internals/ref.mjs");
var _vattrMjs = require("../../lustre/vdom/vattr.mjs");
class Fragment extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, children, keyed_children){
        super();
        this.kind = kind;
        this.key = key;
        this.children = children;
        this.keyed_children = keyed_children;
    }
}
const Element$Fragment = (kind, key, children, keyed_children)=>new Fragment(kind, key, children, keyed_children);
const Element$isFragment = (value)=>value instanceof Fragment;
const Element$Fragment$kind = (value)=>value.kind;
const Element$Fragment$0 = (value)=>value.kind;
const Element$Fragment$key = (value)=>value.key;
const Element$Fragment$1 = (value)=>value.key;
const Element$Fragment$children = (value)=>value.children;
const Element$Fragment$2 = (value)=>value.children;
const Element$Fragment$keyed_children = (value)=>value.keyed_children;
const Element$Fragment$3 = (value)=>value.keyed_children;
class Element extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, namespace, tag, attributes, children, keyed_children, self_closing, void$){
        super();
        this.kind = kind;
        this.key = key;
        this.namespace = namespace;
        this.tag = tag;
        this.attributes = attributes;
        this.children = children;
        this.keyed_children = keyed_children;
        this.self_closing = self_closing;
        this.void = void$;
    }
}
const Element$Element = (kind, key, namespace, tag, attributes, children, keyed_children, self_closing, void$)=>new Element(kind, key, namespace, tag, attributes, children, keyed_children, self_closing, void$);
const Element$isElement = (value)=>value instanceof Element;
const Element$Element$kind = (value)=>value.kind;
const Element$Element$0 = (value)=>value.kind;
const Element$Element$key = (value)=>value.key;
const Element$Element$1 = (value)=>value.key;
const Element$Element$namespace = (value)=>value.namespace;
const Element$Element$2 = (value)=>value.namespace;
const Element$Element$tag = (value)=>value.tag;
const Element$Element$3 = (value)=>value.tag;
const Element$Element$attributes = (value)=>value.attributes;
const Element$Element$4 = (value)=>value.attributes;
const Element$Element$children = (value)=>value.children;
const Element$Element$5 = (value)=>value.children;
const Element$Element$keyed_children = (value)=>value.keyed_children;
const Element$Element$6 = (value)=>value.keyed_children;
const Element$Element$self_closing = (value)=>value.self_closing;
const Element$Element$7 = (value)=>value.self_closing;
const Element$Element$void = (value)=>value.void;
const Element$Element$8 = (value)=>value.void;
class Text extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, content){
        super();
        this.kind = kind;
        this.key = key;
        this.content = content;
    }
}
const Element$Text = (kind, key, content)=>new Text(kind, key, content);
const Element$isText = (value)=>value instanceof Text;
const Element$Text$kind = (value)=>value.kind;
const Element$Text$0 = (value)=>value.kind;
const Element$Text$key = (value)=>value.key;
const Element$Text$1 = (value)=>value.key;
const Element$Text$content = (value)=>value.content;
const Element$Text$2 = (value)=>value.content;
class UnsafeInnerHtml extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, namespace, tag, attributes, inner_html){
        super();
        this.kind = kind;
        this.key = key;
        this.namespace = namespace;
        this.tag = tag;
        this.attributes = attributes;
        this.inner_html = inner_html;
    }
}
const Element$UnsafeInnerHtml = (kind, key, namespace, tag, attributes, inner_html)=>new UnsafeInnerHtml(kind, key, namespace, tag, attributes, inner_html);
const Element$isUnsafeInnerHtml = (value)=>value instanceof UnsafeInnerHtml;
const Element$UnsafeInnerHtml$kind = (value)=>value.kind;
const Element$UnsafeInnerHtml$0 = (value)=>value.kind;
const Element$UnsafeInnerHtml$key = (value)=>value.key;
const Element$UnsafeInnerHtml$1 = (value)=>value.key;
const Element$UnsafeInnerHtml$namespace = (value)=>value.namespace;
const Element$UnsafeInnerHtml$2 = (value)=>value.namespace;
const Element$UnsafeInnerHtml$tag = (value)=>value.tag;
const Element$UnsafeInnerHtml$3 = (value)=>value.tag;
const Element$UnsafeInnerHtml$attributes = (value)=>value.attributes;
const Element$UnsafeInnerHtml$4 = (value)=>value.attributes;
const Element$UnsafeInnerHtml$inner_html = (value)=>value.inner_html;
const Element$UnsafeInnerHtml$5 = (value)=>value.inner_html;
class Map extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, mapper, child){
        super();
        this.kind = kind;
        this.key = key;
        this.mapper = mapper;
        this.child = child;
    }
}
const Element$Map = (kind, key, mapper, child)=>new Map(kind, key, mapper, child);
const Element$isMap = (value)=>value instanceof Map;
const Element$Map$kind = (value)=>value.kind;
const Element$Map$0 = (value)=>value.kind;
const Element$Map$key = (value)=>value.key;
const Element$Map$1 = (value)=>value.key;
const Element$Map$mapper = (value)=>value.mapper;
const Element$Map$2 = (value)=>value.mapper;
const Element$Map$child = (value)=>value.child;
const Element$Map$3 = (value)=>value.child;
class Memo extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, dependencies, view){
        super();
        this.kind = kind;
        this.key = key;
        this.dependencies = dependencies;
        this.view = view;
    }
}
const Element$Memo = (kind, key, dependencies, view)=>new Memo(kind, key, dependencies, view);
const Element$isMemo = (value)=>value instanceof Memo;
const Element$Memo$kind = (value)=>value.kind;
const Element$Memo$0 = (value)=>value.kind;
const Element$Memo$key = (value)=>value.key;
const Element$Memo$1 = (value)=>value.key;
const Element$Memo$dependencies = (value)=>value.dependencies;
const Element$Memo$2 = (value)=>value.dependencies;
const Element$Memo$view = (value)=>value.view;
const Element$Memo$3 = (value)=>value.view;
const Element$key = (value)=>value.key;
const Element$kind = (value)=>value.kind;
const fragment_kind = 0;
const element_kind = 1;
const text_kind = 2;
const unsafe_inner_html_kind = 3;
const map_kind = 4;
const memo_kind = 5;
function is_void_html_element(tag, namespace) {
    if (namespace === "") {
        if (tag === "area") return true;
        else if (tag === "base") return true;
        else if (tag === "br") return true;
        else if (tag === "col") return true;
        else if (tag === "embed") return true;
        else if (tag === "hr") return true;
        else if (tag === "img") return true;
        else if (tag === "input") return true;
        else if (tag === "link") return true;
        else if (tag === "meta") return true;
        else if (tag === "param") return true;
        else if (tag === "source") return true;
        else if (tag === "track") return true;
        else if (tag === "wbr") return true;
        else return false;
    } else return false;
}
function to_keyed(key, node) {
    if (node instanceof Fragment) return new Fragment(node.kind, key, node.children, node.keyed_children);
    else if (node instanceof Element) return new Element(node.kind, key, node.namespace, node.tag, node.attributes, node.children, node.keyed_children, node.self_closing, node.void);
    else if (node instanceof Text) return new Text(node.kind, key, node.content);
    else if (node instanceof UnsafeInnerHtml) return new UnsafeInnerHtml(node.kind, key, node.namespace, node.tag, node.attributes, node.inner_html);
    else if (node instanceof Map) {
        let child = node.child;
        return new Map(node.kind, key, node.mapper, to_keyed(key, child));
    } else {
        let view = node.view;
        return new Memo(node.kind, key, node.dependencies, ()=>{
            return to_keyed(key, view());
        });
    }
}
function text_to_json(kind, key, content) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.string(_pipe$1, "content", content);
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function unsafe_inner_html_to_json(kind, key, namespace, tag, attributes, inner_html) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.string(_pipe$1, "namespace", namespace);
    let _pipe$3 = _jsonObjectBuilderMjs.string(_pipe$2, "tag", tag);
    let _pipe$4 = _jsonObjectBuilderMjs.list(_pipe$3, "attributes", attributes, _vattrMjs.to_json);
    let _pipe$5 = _jsonObjectBuilderMjs.string(_pipe$4, "inner_html", inner_html);
    return _jsonObjectBuilderMjs.build(_pipe$5);
}
function marker_comment(label, key) {
    if (key === "") return _stringTreeMjs.from_string("<!-- " + label + " -->");
    else {
        let _pipe = _stringTreeMjs.from_string("<!-- " + label + " key=\"");
        let _pipe$1 = _stringTreeMjs.append(_pipe, _houdiniMjs.escape(key));
        return _stringTreeMjs.append(_pipe$1, "\" -->");
    }
}
function fragment(key, children, keyed_children) {
    return new Fragment(fragment_kind, key, children, keyed_children);
}
function element(key, namespace, tag, attributes, children, keyed_children, self_closing, void$) {
    return new Element(element_kind, key, namespace, tag, _vattrMjs.prepare(attributes), children, keyed_children, self_closing, void$);
}
function text(key, content) {
    return new Text(text_kind, key, content);
}
function unsafe_inner_html(key, namespace, tag, attributes, inner_html) {
    return new UnsafeInnerHtml(unsafe_inner_html_kind, key, namespace, tag, _vattrMjs.prepare(attributes), inner_html);
}
function map(element, mapper) {
    if (element instanceof Map) {
        let child_mapper = element.mapper;
        return new Map(map_kind, element.key, (handler)=>{
            return (0, _functionMjs.identity)(mapper)(child_mapper(handler));
        }, (0, _functionMjs.identity)(element.child));
    } else return new Map(map_kind, element.key, (0, _functionMjs.identity)(mapper), (0, _functionMjs.identity)(element));
}
function memo(key, dependencies, view) {
    return new Memo(memo_kind, key, dependencies, view);
}
function children_to_snapshot_builder(loop$html, loop$children, loop$raw, loop$debug, loop$namespace, loop$indent) {
    while(true){
        let html = loop$html;
        let children = loop$children;
        let raw = loop$raw;
        let debug = loop$debug;
        let namespace = loop$namespace;
        let indent = loop$indent;
        if (children instanceof (0, _gleamMjs.Empty)) return html;
        else {
            let $ = children.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let child = children.head;
                let rest = $;
                let _pipe = child;
                let _pipe$1 = do_to_snapshot_builder(_pipe, raw, debug, namespace, indent);
                let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
                let _pipe$3 = _stringTreeMjs.prepend_tree(_pipe$2, html);
                loop$html = _pipe$3;
                loop$children = rest;
                loop$raw = raw;
                loop$debug = debug;
                loop$namespace = namespace;
                loop$indent = indent;
            } else {
                let $1 = children.head;
                if ($1 instanceof Text) {
                    let $2 = $.head;
                    if ($2 instanceof Text) {
                        let rest = $.tail;
                        let a = $1.content;
                        let b = $2.content;
                        loop$html = html;
                        loop$children = (0, _gleamMjs.prepend)(new Text(text_kind, "", a + b), rest);
                        loop$raw = raw;
                        loop$debug = debug;
                        loop$namespace = namespace;
                        loop$indent = indent;
                    } else {
                        let child = $1;
                        let rest = $;
                        let _pipe = child;
                        let _pipe$1 = do_to_snapshot_builder(_pipe, raw, debug, namespace, indent);
                        let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
                        let _pipe$3 = _stringTreeMjs.prepend_tree(_pipe$2, html);
                        loop$html = _pipe$3;
                        loop$children = rest;
                        loop$raw = raw;
                        loop$debug = debug;
                        loop$namespace = namespace;
                        loop$indent = indent;
                    }
                } else {
                    let child = $1;
                    let rest = $;
                    let _pipe = child;
                    let _pipe$1 = do_to_snapshot_builder(_pipe, raw, debug, namespace, indent);
                    let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
                    let _pipe$3 = _stringTreeMjs.prepend_tree(_pipe$2, html);
                    loop$html = _pipe$3;
                    loop$children = rest;
                    loop$raw = raw;
                    loop$debug = debug;
                    loop$namespace = namespace;
                    loop$indent = indent;
                }
            }
        }
    }
}
function do_to_snapshot_builder(loop$node, loop$raw, loop$debug, loop$parent_namespace, loop$indent) {
    while(true){
        let node = loop$node;
        let raw = loop$raw;
        let debug = loop$debug;
        let parent_namespace = loop$parent_namespace;
        let indent = loop$indent;
        let spaces = _stringMjs.repeat("  ", indent);
        if (node instanceof Fragment) {
            if (debug) {
                let key = node.key;
                let children = node.children;
                let _pipe = marker_comment("lustre:fragment", key);
                let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
                let _pipe$3 = children_to_snapshot_builder(_pipe$2, children, raw, debug, parent_namespace, indent + 1);
                let _pipe$4 = _stringTreeMjs.append(_pipe$3, spaces);
                return _stringTreeMjs.append_tree(_pipe$4, marker_comment("/lustre:fragment", ""));
            } else {
                let children = node.children;
                return children_to_snapshot_builder(_stringTreeMjs.new$(), children, raw, debug, parent_namespace, indent);
            }
        } else if (node instanceof Element) {
            let $ = node.self_closing;
            if ($) {
                let key = node.key;
                let namespace = node.namespace;
                let tag = node.tag;
                let attributes = node.attributes;
                let html = _stringTreeMjs.from_string("<" + tag);
                let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                let _pipe = html;
                let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                let _pipe$2 = _stringTreeMjs.append_tree(_pipe$1, attributes$1);
                return _stringTreeMjs.append(_pipe$2, "/>");
            } else {
                let $1 = node.void;
                if ($1) {
                    let key = node.key;
                    let namespace = node.namespace;
                    let tag = node.tag;
                    let attributes = node.attributes;
                    let html = _stringTreeMjs.from_string("<" + tag);
                    let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                    let _pipe = html;
                    let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                    let _pipe$2 = _stringTreeMjs.append_tree(_pipe$1, attributes$1);
                    return _stringTreeMjs.append(_pipe$2, ">");
                } else {
                    let $2 = node.children;
                    if ($2 instanceof (0, _gleamMjs.Empty)) {
                        let key = node.key;
                        let namespace = node.namespace;
                        let tag = node.tag;
                        let attributes = node.attributes;
                        let html = _stringTreeMjs.from_string("<" + tag);
                        let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                        let _pipe = html;
                        let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                        let _pipe$2 = _stringTreeMjs.append_tree(_pipe$1, attributes$1);
                        let _pipe$3 = _stringTreeMjs.append(_pipe$2, ">");
                        return _stringTreeMjs.append(_pipe$3, "</" + tag + ">");
                    } else {
                        let key = node.key;
                        let namespace = node.namespace;
                        let tag = node.tag;
                        let attributes = node.attributes;
                        let children = $2;
                        let html = _stringTreeMjs.from_string("<" + tag);
                        let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                        let _pipe = html;
                        let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                        let _pipe$2 = _stringTreeMjs.append_tree(_pipe$1, attributes$1);
                        let _pipe$3 = _stringTreeMjs.append(_pipe$2, ">\n");
                        let _pipe$4 = children_to_snapshot_builder(_pipe$3, children, raw, debug, namespace, indent + 1);
                        let _pipe$5 = _stringTreeMjs.append(_pipe$4, spaces);
                        return _stringTreeMjs.append(_pipe$5, "</" + tag + ">");
                    }
                }
            }
        } else if (node instanceof Text) {
            let $ = node.content;
            if ($ === "") return _stringTreeMjs.new$();
            else if (raw) {
                let content = $;
                return _stringTreeMjs.from_strings((0, _gleamMjs.toList)([
                    spaces,
                    content
                ]));
            } else {
                let content = $;
                return _stringTreeMjs.from_strings((0, _gleamMjs.toList)([
                    spaces,
                    _houdiniMjs.escape(content)
                ]));
            }
        } else if (node instanceof UnsafeInnerHtml) {
            let key = node.key;
            let namespace = node.namespace;
            let tag = node.tag;
            let attributes = node.attributes;
            let inner_html = node.inner_html;
            let html = _stringTreeMjs.from_string("<" + tag);
            let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
            let _pipe = html;
            let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
            let _pipe$2 = _stringTreeMjs.append_tree(_pipe$1, attributes$1);
            let _pipe$3 = _stringTreeMjs.append(_pipe$2, ">");
            let _pipe$4 = _stringTreeMjs.append(_pipe$3, inner_html);
            return _stringTreeMjs.append(_pipe$4, "</" + tag + ">");
        } else if (node instanceof Map) {
            if (debug) {
                let key = node.key;
                let child = node.child;
                let _pipe = marker_comment("lustre:map", key);
                let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
                let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
                return _stringTreeMjs.append_tree(_pipe$2, do_to_snapshot_builder(child, raw, debug, parent_namespace, indent + 1));
            } else {
                let child = node.child;
                loop$node = child;
                loop$raw = raw;
                loop$debug = debug;
                loop$parent_namespace = parent_namespace;
                loop$indent = indent;
            }
        } else if (debug) {
            let key = node.key;
            let view = node.view;
            let _pipe = marker_comment("lustre:memo", key);
            let _pipe$1 = _stringTreeMjs.prepend(_pipe, spaces);
            let _pipe$2 = _stringTreeMjs.append(_pipe$1, "\n");
            return _stringTreeMjs.append_tree(_pipe$2, do_to_snapshot_builder(view(), raw, debug, parent_namespace, indent + 1));
        } else {
            let view = node.view;
            loop$node = view();
            loop$raw = raw;
            loop$debug = debug;
            loop$parent_namespace = parent_namespace;
            loop$indent = indent;
        }
    }
}
function to_snapshot(node, debug) {
    let _pipe = do_to_snapshot_builder(node, false, debug, "", 0);
    return _stringTreeMjs.to_string(_pipe);
}
function children_to_string_tree(html, children, namespace) {
    return _listMjs.fold(children, html, (html, child)=>{
        return _stringTreeMjs.append_tree(html, to_string_tree(child, namespace));
    });
}
function to_string_tree(node, parent_namespace) {
    if (node instanceof Fragment) {
        let key = node.key;
        let children = node.children;
        let _pipe = marker_comment("lustre:fragment", key);
        let _pipe$1 = children_to_string_tree(_pipe, children, parent_namespace);
        return _stringTreeMjs.append_tree(_pipe$1, marker_comment("/lustre:fragment", ""));
    } else if (node instanceof Element) {
        let self_closing = node.self_closing;
        if (self_closing) {
            let key = node.key;
            let namespace = node.namespace;
            let tag = node.tag;
            let attributes = node.attributes;
            let html = _stringTreeMjs.from_string("<" + tag);
            let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
            let _pipe = html;
            let _pipe$1 = _stringTreeMjs.append_tree(_pipe, attributes$1);
            return _stringTreeMjs.append(_pipe$1, "/>");
        } else {
            let void$ = node.void;
            if (void$) {
                let key = node.key;
                let namespace = node.namespace;
                let tag = node.tag;
                let attributes = node.attributes;
                let html = _stringTreeMjs.from_string("<" + tag);
                let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                let _pipe = html;
                let _pipe$1 = _stringTreeMjs.append_tree(_pipe, attributes$1);
                return _stringTreeMjs.append(_pipe$1, ">");
            } else {
                let key = node.key;
                let namespace = node.namespace;
                let tag = node.tag;
                let attributes = node.attributes;
                let children = node.children;
                let html = _stringTreeMjs.from_string("<" + tag);
                let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
                let _pipe = html;
                let _pipe$1 = _stringTreeMjs.append_tree(_pipe, attributes$1);
                let _pipe$2 = _stringTreeMjs.append(_pipe$1, ">");
                let _pipe$3 = children_to_string_tree(_pipe$2, children, namespace);
                return _stringTreeMjs.append(_pipe$3, "</" + tag + ">");
            }
        }
    } else if (node instanceof Text) {
        let $ = node.content;
        if ($ === "") return _stringTreeMjs.new$();
        else {
            let content = $;
            return _stringTreeMjs.from_string(_houdiniMjs.escape(content));
        }
    } else if (node instanceof UnsafeInnerHtml) {
        let key = node.key;
        let namespace = node.namespace;
        let tag = node.tag;
        let attributes = node.attributes;
        let inner_html = node.inner_html;
        let html = _stringTreeMjs.from_string("<" + tag);
        let attributes$1 = _vattrMjs.to_string_tree(key, namespace, parent_namespace, attributes);
        let _pipe = html;
        let _pipe$1 = _stringTreeMjs.append_tree(_pipe, attributes$1);
        let _pipe$2 = _stringTreeMjs.append(_pipe$1, ">");
        let _pipe$3 = _stringTreeMjs.append(_pipe$2, inner_html);
        return _stringTreeMjs.append(_pipe$3, "</" + tag + ">");
    } else if (node instanceof Map) {
        let key = node.key;
        let child = node.child;
        let _pipe = marker_comment("lustre:map", key);
        return _stringTreeMjs.append_tree(_pipe, to_string_tree(child, parent_namespace));
    } else {
        let key = node.key;
        let view = node.view;
        let _pipe = marker_comment("lustre:memo", key);
        return _stringTreeMjs.append_tree(_pipe, to_string_tree(view(), parent_namespace));
    }
}
function to_string(node) {
    let _pipe = node;
    let _pipe$1 = to_string_tree(_pipe, "");
    return _stringTreeMjs.to_string(_pipe$1);
}
function fragment_to_json(kind, key, children, memos) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.list(_pipe$1, "children", children, (_capture)=>{
        return to_json(_capture, memos);
    });
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function to_json(node, memos) {
    if (node instanceof Fragment) {
        let kind = node.kind;
        let key = node.key;
        let children = node.children;
        return fragment_to_json(kind, key, children, memos);
    } else if (node instanceof Element) {
        let kind = node.kind;
        let key = node.key;
        let namespace = node.namespace;
        let tag = node.tag;
        let attributes = node.attributes;
        let children = node.children;
        return element_to_json(kind, key, namespace, tag, attributes, children, memos);
    } else if (node instanceof Text) {
        let kind = node.kind;
        let key = node.key;
        let content = node.content;
        return text_to_json(kind, key, content);
    } else if (node instanceof UnsafeInnerHtml) {
        let kind = node.kind;
        let key = node.key;
        let namespace = node.namespace;
        let tag = node.tag;
        let attributes = node.attributes;
        let inner_html = node.inner_html;
        return unsafe_inner_html_to_json(kind, key, namespace, tag, attributes, inner_html);
    } else if (node instanceof Map) {
        let kind = node.kind;
        let key = node.key;
        let child = node.child;
        return map_to_json(kind, key, child, memos);
    } else {
        let view = node.view;
        return memo_to_json(view, memos);
    }
}
function element_to_json(kind, key, namespace, tag, attributes, children, memos) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.string(_pipe$1, "namespace", namespace);
    let _pipe$3 = _jsonObjectBuilderMjs.string(_pipe$2, "tag", tag);
    let _pipe$4 = _jsonObjectBuilderMjs.list(_pipe$3, "attributes", attributes, _vattrMjs.to_json);
    let _pipe$5 = _jsonObjectBuilderMjs.list(_pipe$4, "children", children, (_capture)=>{
        return to_json(_capture, memos);
    });
    return _jsonObjectBuilderMjs.build(_pipe$5);
}
function memo_to_json(view, memos) {
    let child = _mutableMapMjs.get_or_compute(memos, view, view);
    return to_json(child, memos);
}
function map_to_json(kind, key, child, memos) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.json(_pipe$1, "child", to_json(child, memos));
    return _jsonObjectBuilderMjs.build(_pipe$2);
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/function.mjs":"2jh6y","../../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../../gleam_stdlib/gleam/string_tree.mjs":"8IH0o","../../../houdini/houdini.mjs":"e94ou","../../gleam.mjs":"jNPQG","../../lustre/internals/json_object_builder.mjs":"31ZqD","../../lustre/internals/mutable_map.mjs":"6NvMa","../../lustre/internals/ref.mjs":"gnct2","../../lustre/vdom/vattr.mjs":"jrrcC","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eLT3l":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 *
 */ parcelHelpers.export(exports, "html", ()=>html);
parcelHelpers.export(exports, "text", ()=>text);
/**
 *
 */ parcelHelpers.export(exports, "base", ()=>base);
/**
 *
 */ parcelHelpers.export(exports, "head", ()=>head);
/**
 *
 */ parcelHelpers.export(exports, "link", ()=>link);
/**
 *
 */ parcelHelpers.export(exports, "meta", ()=>meta);
/**
 *
 */ parcelHelpers.export(exports, "style", ()=>style);
/**
 *
 */ parcelHelpers.export(exports, "title", ()=>title);
/**
 *
 */ parcelHelpers.export(exports, "body", ()=>body);
/**
 *
 */ parcelHelpers.export(exports, "address", ()=>address);
/**
 *
 */ parcelHelpers.export(exports, "article", ()=>article);
/**
 *
 */ parcelHelpers.export(exports, "aside", ()=>aside);
/**
 *
 */ parcelHelpers.export(exports, "footer", ()=>footer);
/**
 *
 */ parcelHelpers.export(exports, "header", ()=>header);
/**
 *
 */ parcelHelpers.export(exports, "h1", ()=>h1);
/**
 *
 */ parcelHelpers.export(exports, "h2", ()=>h2);
/**
 *
 */ parcelHelpers.export(exports, "h3", ()=>h3);
/**
 *
 */ parcelHelpers.export(exports, "h4", ()=>h4);
/**
 *
 */ parcelHelpers.export(exports, "h5", ()=>h5);
/**
 *
 */ parcelHelpers.export(exports, "h6", ()=>h6);
/**
 *
 */ parcelHelpers.export(exports, "hgroup", ()=>hgroup);
/**
 *
 */ parcelHelpers.export(exports, "main", ()=>main);
/**
 *
 */ parcelHelpers.export(exports, "nav", ()=>nav);
/**
 *
 */ parcelHelpers.export(exports, "section", ()=>section);
/**
 *
 */ parcelHelpers.export(exports, "search", ()=>search);
/**
 *
 */ parcelHelpers.export(exports, "blockquote", ()=>blockquote);
/**
 *
 */ parcelHelpers.export(exports, "dd", ()=>dd);
/**
 *
 */ parcelHelpers.export(exports, "div", ()=>div);
/**
 *
 */ parcelHelpers.export(exports, "dl", ()=>dl);
/**
 *
 */ parcelHelpers.export(exports, "dt", ()=>dt);
/**
 *
 */ parcelHelpers.export(exports, "figcaption", ()=>figcaption);
/**
 *
 */ parcelHelpers.export(exports, "figure", ()=>figure);
/**
 *
 */ parcelHelpers.export(exports, "hr", ()=>hr);
/**
 *
 */ parcelHelpers.export(exports, "li", ()=>li);
/**
 *
 */ parcelHelpers.export(exports, "menu", ()=>menu);
/**
 *
 */ parcelHelpers.export(exports, "ol", ()=>ol);
/**
 *
 */ parcelHelpers.export(exports, "p", ()=>p);
/**
 *
 */ parcelHelpers.export(exports, "pre", ()=>pre);
/**
 *
 */ parcelHelpers.export(exports, "ul", ()=>ul);
/**
 *
 */ parcelHelpers.export(exports, "a", ()=>a);
/**
 *
 */ parcelHelpers.export(exports, "abbr", ()=>abbr);
/**
 *
 */ parcelHelpers.export(exports, "b", ()=>b);
/**
 *
 */ parcelHelpers.export(exports, "bdi", ()=>bdi);
/**
 *
 */ parcelHelpers.export(exports, "bdo", ()=>bdo);
/**
 *
 */ parcelHelpers.export(exports, "br", ()=>br);
/**
 *
 */ parcelHelpers.export(exports, "cite", ()=>cite);
/**
 *
 */ parcelHelpers.export(exports, "code", ()=>code);
/**
 *
 */ parcelHelpers.export(exports, "data", ()=>data);
/**
 *
 */ parcelHelpers.export(exports, "dfn", ()=>dfn);
/**
 *
 */ parcelHelpers.export(exports, "em", ()=>em);
/**
 *
 */ parcelHelpers.export(exports, "i", ()=>i);
/**
 *
 */ parcelHelpers.export(exports, "kbd", ()=>kbd);
/**
 *
 */ parcelHelpers.export(exports, "mark", ()=>mark);
/**
 *
 */ parcelHelpers.export(exports, "q", ()=>q);
/**
 *
 */ parcelHelpers.export(exports, "rp", ()=>rp);
/**
 *
 */ parcelHelpers.export(exports, "rt", ()=>rt);
/**
 *
 */ parcelHelpers.export(exports, "ruby", ()=>ruby);
/**
 *
 */ parcelHelpers.export(exports, "s", ()=>s);
/**
 *
 */ parcelHelpers.export(exports, "samp", ()=>samp);
/**
 *
 */ parcelHelpers.export(exports, "small", ()=>small);
/**
 *
 */ parcelHelpers.export(exports, "span", ()=>span);
/**
 *
 */ parcelHelpers.export(exports, "strong", ()=>strong);
/**
 *
 */ parcelHelpers.export(exports, "sub", ()=>sub);
/**
 *
 */ parcelHelpers.export(exports, "sup", ()=>sup);
/**
 *
 */ parcelHelpers.export(exports, "time", ()=>time);
/**
 *
 */ parcelHelpers.export(exports, "u", ()=>u);
/**
 *
 */ parcelHelpers.export(exports, "var$", ()=>var$);
/**
 *
 */ parcelHelpers.export(exports, "wbr", ()=>wbr);
/**
 *
 */ parcelHelpers.export(exports, "area", ()=>area);
/**
 *
 */ parcelHelpers.export(exports, "audio", ()=>audio);
/**
 *
 */ parcelHelpers.export(exports, "img", ()=>img);
/**
 * Used with <area> elements to define an image map (a clickable link area).
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 *
 */ parcelHelpers.export(exports, "track", ()=>track);
/**
 *
 */ parcelHelpers.export(exports, "video", ()=>video);
/**
 *
 */ parcelHelpers.export(exports, "embed", ()=>embed);
/**
 *
 */ parcelHelpers.export(exports, "iframe", ()=>iframe);
/**
 *
 */ parcelHelpers.export(exports, "object", ()=>object);
/**
 *
 */ parcelHelpers.export(exports, "picture", ()=>picture);
/**
 *
 */ parcelHelpers.export(exports, "portal", ()=>portal);
/**
 *
 */ parcelHelpers.export(exports, "source", ()=>source);
/**
 *
 */ parcelHelpers.export(exports, "math", ()=>math);
/**
 *
 */ parcelHelpers.export(exports, "svg", ()=>svg);
/**
 *
 */ parcelHelpers.export(exports, "canvas", ()=>canvas);
/**
 *
 */ parcelHelpers.export(exports, "noscript", ()=>noscript);
/**
 *
 */ parcelHelpers.export(exports, "script", ()=>script);
/**
 *
 */ parcelHelpers.export(exports, "del", ()=>del);
/**
 *
 */ parcelHelpers.export(exports, "ins", ()=>ins);
/**
 *
 */ parcelHelpers.export(exports, "caption", ()=>caption);
/**
 *
 */ parcelHelpers.export(exports, "col", ()=>col);
/**
 *
 */ parcelHelpers.export(exports, "colgroup", ()=>colgroup);
/**
 *
 */ parcelHelpers.export(exports, "table", ()=>table);
/**
 *
 */ parcelHelpers.export(exports, "tbody", ()=>tbody);
/**
 *
 */ parcelHelpers.export(exports, "td", ()=>td);
/**
 *
 */ parcelHelpers.export(exports, "tfoot", ()=>tfoot);
/**
 *
 */ parcelHelpers.export(exports, "th", ()=>th);
/**
 *
 */ parcelHelpers.export(exports, "thead", ()=>thead);
/**
 *
 */ parcelHelpers.export(exports, "tr", ()=>tr);
/**
 *
 */ parcelHelpers.export(exports, "button", ()=>button);
/**
 *
 */ parcelHelpers.export(exports, "datalist", ()=>datalist);
/**
 *
 */ parcelHelpers.export(exports, "fieldset", ()=>fieldset);
/**
 *
 */ parcelHelpers.export(exports, "form", ()=>form);
/**
 *
 */ parcelHelpers.export(exports, "input", ()=>input);
/**
 *
 */ parcelHelpers.export(exports, "label", ()=>label);
/**
 *
 */ parcelHelpers.export(exports, "legend", ()=>legend);
/**
 *
 */ parcelHelpers.export(exports, "meter", ()=>meter);
/**
 *
 */ parcelHelpers.export(exports, "optgroup", ()=>optgroup);
/**
 *
 */ parcelHelpers.export(exports, "option", ()=>option);
/**
 *
 */ parcelHelpers.export(exports, "output", ()=>output);
/**
 *
 */ parcelHelpers.export(exports, "progress", ()=>progress);
/**
 *
 */ parcelHelpers.export(exports, "select", ()=>select);
/**
 *
 */ parcelHelpers.export(exports, "textarea", ()=>textarea);
/**
 *
 */ parcelHelpers.export(exports, "details", ()=>details);
/**
 *
 */ parcelHelpers.export(exports, "dialog", ()=>dialog);
/**
 *
 */ parcelHelpers.export(exports, "summary", ()=>summary);
/**
 *
 */ parcelHelpers.export(exports, "slot", ()=>slot);
/**
 *
 */ parcelHelpers.export(exports, "template", ()=>template);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _attributeMjs = require("../../lustre/attribute.mjs");
var _elementMjs = require("../../lustre/element.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
function html(attrs, children) {
    return (0, _elementMjs.element)("html", attrs, children);
}
function text(content) {
    return _elementMjs.text(content);
}
function base(attrs) {
    return (0, _elementMjs.element)("base", attrs, _constantsMjs.empty_list);
}
function head(attrs, children) {
    return (0, _elementMjs.element)("head", attrs, children);
}
function link(attrs) {
    return (0, _elementMjs.element)("link", attrs, _constantsMjs.empty_list);
}
function meta(attrs) {
    return (0, _elementMjs.element)("meta", attrs, _constantsMjs.empty_list);
}
function style(attrs, css) {
    return _elementMjs.unsafe_raw_html("", "style", attrs, css);
}
function title(attrs, content) {
    return (0, _elementMjs.element)("title", attrs, (0, _gleamMjs.toList)([
        text(content)
    ]));
}
function body(attrs, children) {
    return (0, _elementMjs.element)("body", attrs, children);
}
function address(attrs, children) {
    return (0, _elementMjs.element)("address", attrs, children);
}
function article(attrs, children) {
    return (0, _elementMjs.element)("article", attrs, children);
}
function aside(attrs, children) {
    return (0, _elementMjs.element)("aside", attrs, children);
}
function footer(attrs, children) {
    return (0, _elementMjs.element)("footer", attrs, children);
}
function header(attrs, children) {
    return (0, _elementMjs.element)("header", attrs, children);
}
function h1(attrs, children) {
    return (0, _elementMjs.element)("h1", attrs, children);
}
function h2(attrs, children) {
    return (0, _elementMjs.element)("h2", attrs, children);
}
function h3(attrs, children) {
    return (0, _elementMjs.element)("h3", attrs, children);
}
function h4(attrs, children) {
    return (0, _elementMjs.element)("h4", attrs, children);
}
function h5(attrs, children) {
    return (0, _elementMjs.element)("h5", attrs, children);
}
function h6(attrs, children) {
    return (0, _elementMjs.element)("h6", attrs, children);
}
function hgroup(attrs, children) {
    return (0, _elementMjs.element)("hgroup", attrs, children);
}
function main(attrs, children) {
    return (0, _elementMjs.element)("main", attrs, children);
}
function nav(attrs, children) {
    return (0, _elementMjs.element)("nav", attrs, children);
}
function section(attrs, children) {
    return (0, _elementMjs.element)("section", attrs, children);
}
function search(attrs, children) {
    return (0, _elementMjs.element)("search", attrs, children);
}
function blockquote(attrs, children) {
    return (0, _elementMjs.element)("blockquote", attrs, children);
}
function dd(attrs, children) {
    return (0, _elementMjs.element)("dd", attrs, children);
}
function div(attrs, children) {
    return (0, _elementMjs.element)("div", attrs, children);
}
function dl(attrs, children) {
    return (0, _elementMjs.element)("dl", attrs, children);
}
function dt(attrs, children) {
    return (0, _elementMjs.element)("dt", attrs, children);
}
function figcaption(attrs, children) {
    return (0, _elementMjs.element)("figcaption", attrs, children);
}
function figure(attrs, children) {
    return (0, _elementMjs.element)("figure", attrs, children);
}
function hr(attrs) {
    return (0, _elementMjs.element)("hr", attrs, _constantsMjs.empty_list);
}
function li(attrs, children) {
    return (0, _elementMjs.element)("li", attrs, children);
}
function menu(attrs, children) {
    return (0, _elementMjs.element)("menu", attrs, children);
}
function ol(attrs, children) {
    return (0, _elementMjs.element)("ol", attrs, children);
}
function p(attrs, children) {
    return (0, _elementMjs.element)("p", attrs, children);
}
function pre(attrs, children) {
    return (0, _elementMjs.element)("pre", attrs, children);
}
function ul(attrs, children) {
    return (0, _elementMjs.element)("ul", attrs, children);
}
function a(attrs, children) {
    return (0, _elementMjs.element)("a", attrs, children);
}
function abbr(attrs, children) {
    return (0, _elementMjs.element)("abbr", attrs, children);
}
function b(attrs, children) {
    return (0, _elementMjs.element)("b", attrs, children);
}
function bdi(attrs, children) {
    return (0, _elementMjs.element)("bdi", attrs, children);
}
function bdo(attrs, children) {
    return (0, _elementMjs.element)("bdo", attrs, children);
}
function br(attrs) {
    return (0, _elementMjs.element)("br", attrs, _constantsMjs.empty_list);
}
function cite(attrs, children) {
    return (0, _elementMjs.element)("cite", attrs, children);
}
function code(attrs, children) {
    return (0, _elementMjs.element)("code", attrs, children);
}
function data(attrs, children) {
    return (0, _elementMjs.element)("data", attrs, children);
}
function dfn(attrs, children) {
    return (0, _elementMjs.element)("dfn", attrs, children);
}
function em(attrs, children) {
    return (0, _elementMjs.element)("em", attrs, children);
}
function i(attrs, children) {
    return (0, _elementMjs.element)("i", attrs, children);
}
function kbd(attrs, children) {
    return (0, _elementMjs.element)("kbd", attrs, children);
}
function mark(attrs, children) {
    return (0, _elementMjs.element)("mark", attrs, children);
}
function q(attrs, children) {
    return (0, _elementMjs.element)("q", attrs, children);
}
function rp(attrs, children) {
    return (0, _elementMjs.element)("rp", attrs, children);
}
function rt(attrs, children) {
    return (0, _elementMjs.element)("rt", attrs, children);
}
function ruby(attrs, children) {
    return (0, _elementMjs.element)("ruby", attrs, children);
}
function s(attrs, children) {
    return (0, _elementMjs.element)("s", attrs, children);
}
function samp(attrs, children) {
    return (0, _elementMjs.element)("samp", attrs, children);
}
function small(attrs, children) {
    return (0, _elementMjs.element)("small", attrs, children);
}
function span(attrs, children) {
    return (0, _elementMjs.element)("span", attrs, children);
}
function strong(attrs, children) {
    return (0, _elementMjs.element)("strong", attrs, children);
}
function sub(attrs, children) {
    return (0, _elementMjs.element)("sub", attrs, children);
}
function sup(attrs, children) {
    return (0, _elementMjs.element)("sup", attrs, children);
}
function time(attrs, children) {
    return (0, _elementMjs.element)("time", attrs, children);
}
function u(attrs, children) {
    return (0, _elementMjs.element)("u", attrs, children);
}
function var$(attrs, children) {
    return (0, _elementMjs.element)("var", attrs, children);
}
function wbr(attrs) {
    return (0, _elementMjs.element)("wbr", attrs, _constantsMjs.empty_list);
}
function area(attrs) {
    return (0, _elementMjs.element)("area", attrs, _constantsMjs.empty_list);
}
function audio(attrs, children) {
    return (0, _elementMjs.element)("audio", attrs, children);
}
function img(attrs) {
    return (0, _elementMjs.element)("img", attrs, _constantsMjs.empty_list);
}
function map(attrs, children) {
    return (0, _elementMjs.element)("map", attrs, children);
}
function track(attrs) {
    return (0, _elementMjs.element)("track", attrs, _constantsMjs.empty_list);
}
function video(attrs, children) {
    return (0, _elementMjs.element)("video", attrs, children);
}
function embed(attrs) {
    return (0, _elementMjs.element)("embed", attrs, _constantsMjs.empty_list);
}
function iframe(attrs) {
    return (0, _elementMjs.element)("iframe", attrs, _constantsMjs.empty_list);
}
function object(attrs) {
    return (0, _elementMjs.element)("object", attrs, _constantsMjs.empty_list);
}
function picture(attrs, children) {
    return (0, _elementMjs.element)("picture", attrs, children);
}
function portal(attrs) {
    return (0, _elementMjs.element)("portal", attrs, _constantsMjs.empty_list);
}
function source(attrs) {
    return (0, _elementMjs.element)("source", attrs, _constantsMjs.empty_list);
}
function math(attrs, children) {
    return (0, _elementMjs.namespaced)("http://www.w3.org/1998/Math/MathML", "math", attrs, children);
}
function svg(attrs, children) {
    return (0, _elementMjs.namespaced)("http://www.w3.org/2000/svg", "svg", attrs, children);
}
function canvas(attrs) {
    return (0, _elementMjs.element)("canvas", attrs, _constantsMjs.empty_list);
}
function noscript(attrs, children) {
    return (0, _elementMjs.element)("noscript", attrs, children);
}
function script(attrs, js) {
    return _elementMjs.unsafe_raw_html("", "script", attrs, js);
}
function del(attrs, children) {
    return _elementMjs.element("del", attrs, children);
}
function ins(attrs, children) {
    return _elementMjs.element("ins", attrs, children);
}
function caption(attrs, children) {
    return _elementMjs.element("caption", attrs, children);
}
function col(attrs) {
    return _elementMjs.element("col", attrs, _constantsMjs.empty_list);
}
function colgroup(attrs, children) {
    return _elementMjs.element("colgroup", attrs, children);
}
function table(attrs, children) {
    return _elementMjs.element("table", attrs, children);
}
function tbody(attrs, children) {
    return _elementMjs.element("tbody", attrs, children);
}
function td(attrs, children) {
    return _elementMjs.element("td", attrs, children);
}
function tfoot(attrs, children) {
    return _elementMjs.element("tfoot", attrs, children);
}
function th(attrs, children) {
    return _elementMjs.element("th", attrs, children);
}
function thead(attrs, children) {
    return _elementMjs.element("thead", attrs, children);
}
function tr(attrs, children) {
    return _elementMjs.element("tr", attrs, children);
}
function button(attrs, children) {
    return _elementMjs.element("button", attrs, children);
}
function datalist(attrs, children) {
    return _elementMjs.element("datalist", attrs, children);
}
function fieldset(attrs, children) {
    return _elementMjs.element("fieldset", attrs, children);
}
function form(attrs, children) {
    return _elementMjs.element("form", attrs, children);
}
function input(attrs) {
    return _elementMjs.element("input", attrs, _constantsMjs.empty_list);
}
function label(attrs, children) {
    return _elementMjs.element("label", attrs, children);
}
function legend(attrs, children) {
    return _elementMjs.element("legend", attrs, children);
}
function meter(attrs, children) {
    return _elementMjs.element("meter", attrs, children);
}
function optgroup(attrs, children) {
    return _elementMjs.element("optgroup", attrs, children);
}
function option(attrs, label) {
    return _elementMjs.element("option", attrs, (0, _gleamMjs.toList)([
        _elementMjs.text(label)
    ]));
}
function output(attrs, children) {
    return _elementMjs.element("output", attrs, children);
}
function progress(attrs, children) {
    return _elementMjs.element("progress", attrs, children);
}
function select(attrs, children) {
    return _elementMjs.element("select", attrs, children);
}
function textarea(attrs, content) {
    return _elementMjs.element("textarea", (0, _gleamMjs.prepend)(_attributeMjs.property("value", _jsonMjs.string(content)), attrs), (0, _gleamMjs.toList)([
        _elementMjs.text(content)
    ]));
}
function details(attrs, children) {
    return _elementMjs.element("details", attrs, children);
}
function dialog(attrs, children) {
    return _elementMjs.element("dialog", attrs, children);
}
function summary(attrs, children) {
    return _elementMjs.element("summary", attrs, children);
}
function slot(attrs, fallback) {
    return _elementMjs.element("slot", attrs, fallback);
}
function template(attrs, children) {
    return _elementMjs.element("template", attrs, children);
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam.mjs":"jNPQG","../../lustre/attribute.mjs":"faRXj","../../lustre/element.mjs":"2XxJ4","../../lustre/internals/constants.mjs":"gKFR6","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"fnyl8":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "App", ()=>App);
parcelHelpers.export(exports, "App$App", ()=>App$App);
parcelHelpers.export(exports, "App$isApp", ()=>App$isApp);
parcelHelpers.export(exports, "App$App$name", ()=>App$App$name);
parcelHelpers.export(exports, "App$App$0", ()=>App$App$0);
parcelHelpers.export(exports, "App$App$init", ()=>App$App$init);
parcelHelpers.export(exports, "App$App$1", ()=>App$App$1);
parcelHelpers.export(exports, "App$App$update", ()=>App$App$update);
parcelHelpers.export(exports, "App$App$2", ()=>App$App$2);
parcelHelpers.export(exports, "App$App$view", ()=>App$App$view);
parcelHelpers.export(exports, "App$App$3", ()=>App$App$3);
parcelHelpers.export(exports, "App$App$config", ()=>App$App$config);
parcelHelpers.export(exports, "App$App$4", ()=>App$App$4);
parcelHelpers.export(exports, "Config", ()=>Config);
parcelHelpers.export(exports, "Config$Config", ()=>Config$Config);
parcelHelpers.export(exports, "Config$isConfig", ()=>Config$isConfig);
parcelHelpers.export(exports, "Config$Config$open_shadow_root", ()=>Config$Config$open_shadow_root);
parcelHelpers.export(exports, "Config$Config$0", ()=>Config$Config$0);
parcelHelpers.export(exports, "Config$Config$adopt_styles", ()=>Config$Config$adopt_styles);
parcelHelpers.export(exports, "Config$Config$1", ()=>Config$Config$1);
parcelHelpers.export(exports, "Config$Config$delegates_focus", ()=>Config$Config$delegates_focus);
parcelHelpers.export(exports, "Config$Config$2", ()=>Config$Config$2);
parcelHelpers.export(exports, "Config$Config$attributes", ()=>Config$Config$attributes);
parcelHelpers.export(exports, "Config$Config$3", ()=>Config$Config$3);
parcelHelpers.export(exports, "Config$Config$properties", ()=>Config$Config$properties);
parcelHelpers.export(exports, "Config$Config$4", ()=>Config$Config$4);
parcelHelpers.export(exports, "Config$Config$contexts", ()=>Config$Config$contexts);
parcelHelpers.export(exports, "Config$Config$5", ()=>Config$Config$5);
parcelHelpers.export(exports, "Config$Config$is_form_associated", ()=>Config$Config$is_form_associated);
parcelHelpers.export(exports, "Config$Config$6", ()=>Config$Config$6);
parcelHelpers.export(exports, "Config$Config$on_form_autofill", ()=>Config$Config$on_form_autofill);
parcelHelpers.export(exports, "Config$Config$7", ()=>Config$Config$7);
parcelHelpers.export(exports, "Config$Config$on_form_reset", ()=>Config$Config$on_form_reset);
parcelHelpers.export(exports, "Config$Config$8", ()=>Config$Config$8);
parcelHelpers.export(exports, "Config$Config$on_form_restore", ()=>Config$Config$on_form_restore);
parcelHelpers.export(exports, "Config$Config$9", ()=>Config$Config$9);
parcelHelpers.export(exports, "Config$Config$on_connect", ()=>Config$Config$on_connect);
parcelHelpers.export(exports, "Config$Config$10", ()=>Config$Config$10);
parcelHelpers.export(exports, "Config$Config$on_adopt", ()=>Config$Config$on_adopt);
parcelHelpers.export(exports, "Config$Config$11", ()=>Config$Config$11);
parcelHelpers.export(exports, "Config$Config$on_disconnect", ()=>Config$Config$on_disconnect);
parcelHelpers.export(exports, "Config$Config$12", ()=>Config$Config$12);
parcelHelpers.export(exports, "Option", ()=>Option);
parcelHelpers.export(exports, "Option$Option", ()=>Option$Option);
parcelHelpers.export(exports, "Option$isOption", ()=>Option$isOption);
parcelHelpers.export(exports, "Option$Option$apply", ()=>Option$Option$apply);
parcelHelpers.export(exports, "Option$Option$0", ()=>Option$Option$0);
parcelHelpers.export(exports, "default_config", ()=>default_config);
parcelHelpers.export(exports, "configure_server_component", ()=>configure_server_component);
parcelHelpers.export(exports, "configure", ()=>configure);
var _processMjs = require("../../../gleam_erlang/gleam/erlang/process.mjs");
var _dictMjs = require("../../../gleam_stdlib/gleam/dict.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
var _optionMjs = require("../../../gleam_stdlib/gleam/option.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _effectMjs = require("../../lustre/effect.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
var _runtimeMjs = require("../../lustre/runtime/server/runtime.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
class App extends (0, _gleamMjs.CustomType) {
    constructor(name, init, update, view, config){
        super();
        this.name = name;
        this.init = init;
        this.update = update;
        this.view = view;
        this.config = config;
    }
}
const App$App = (name, init, update, view, config)=>new App(name, init, update, view, config);
const App$isApp = (value)=>value instanceof App;
const App$App$name = (value)=>value.name;
const App$App$0 = (value)=>value.name;
const App$App$init = (value)=>value.init;
const App$App$1 = (value)=>value.init;
const App$App$update = (value)=>value.update;
const App$App$2 = (value)=>value.update;
const App$App$view = (value)=>value.view;
const App$App$3 = (value)=>value.view;
const App$App$config = (value)=>value.config;
const App$App$4 = (value)=>value.config;
class Config extends (0, _gleamMjs.CustomType) {
    constructor(open_shadow_root, adopt_styles, delegates_focus, attributes, properties, contexts, is_form_associated, on_form_autofill, on_form_reset, on_form_restore, on_connect, on_adopt, on_disconnect){
        super();
        this.open_shadow_root = open_shadow_root;
        this.adopt_styles = adopt_styles;
        this.delegates_focus = delegates_focus;
        this.attributes = attributes;
        this.properties = properties;
        this.contexts = contexts;
        this.is_form_associated = is_form_associated;
        this.on_form_autofill = on_form_autofill;
        this.on_form_reset = on_form_reset;
        this.on_form_restore = on_form_restore;
        this.on_connect = on_connect;
        this.on_adopt = on_adopt;
        this.on_disconnect = on_disconnect;
    }
}
const Config$Config = (open_shadow_root, adopt_styles, delegates_focus, attributes, properties, contexts, is_form_associated, on_form_autofill, on_form_reset, on_form_restore, on_connect, on_adopt, on_disconnect)=>new Config(open_shadow_root, adopt_styles, delegates_focus, attributes, properties, contexts, is_form_associated, on_form_autofill, on_form_reset, on_form_restore, on_connect, on_adopt, on_disconnect);
const Config$isConfig = (value)=>value instanceof Config;
const Config$Config$open_shadow_root = (value)=>value.open_shadow_root;
const Config$Config$0 = (value)=>value.open_shadow_root;
const Config$Config$adopt_styles = (value)=>value.adopt_styles;
const Config$Config$1 = (value)=>value.adopt_styles;
const Config$Config$delegates_focus = (value)=>value.delegates_focus;
const Config$Config$2 = (value)=>value.delegates_focus;
const Config$Config$attributes = (value)=>value.attributes;
const Config$Config$3 = (value)=>value.attributes;
const Config$Config$properties = (value)=>value.properties;
const Config$Config$4 = (value)=>value.properties;
const Config$Config$contexts = (value)=>value.contexts;
const Config$Config$5 = (value)=>value.contexts;
const Config$Config$is_form_associated = (value)=>value.is_form_associated;
const Config$Config$6 = (value)=>value.is_form_associated;
const Config$Config$on_form_autofill = (value)=>value.on_form_autofill;
const Config$Config$7 = (value)=>value.on_form_autofill;
const Config$Config$on_form_reset = (value)=>value.on_form_reset;
const Config$Config$8 = (value)=>value.on_form_reset;
const Config$Config$on_form_restore = (value)=>value.on_form_restore;
const Config$Config$9 = (value)=>value.on_form_restore;
const Config$Config$on_connect = (value)=>value.on_connect;
const Config$Config$10 = (value)=>value.on_connect;
const Config$Config$on_adopt = (value)=>value.on_adopt;
const Config$Config$11 = (value)=>value.on_adopt;
const Config$Config$on_disconnect = (value)=>value.on_disconnect;
const Config$Config$12 = (value)=>value.on_disconnect;
class Option extends (0, _gleamMjs.CustomType) {
    constructor(apply){
        super();
        this.apply = apply;
    }
}
const Option$Option = (apply)=>new Option(apply);
const Option$isOption = (value)=>value instanceof Option;
const Option$Option$apply = (value)=>value.apply;
const Option$Option$0 = (value)=>value.apply;
const default_config = /* @__PURE__ */ new Config(true, true, false, _constantsMjs.empty_list, _constantsMjs.empty_list, _constantsMjs.empty_list, false, /* @__PURE__ */ new _optionMjs.None(), /* @__PURE__ */ new _optionMjs.None(), /* @__PURE__ */ new _optionMjs.None(), /* @__PURE__ */ new _optionMjs.None(), /* @__PURE__ */ new _optionMjs.None(), /* @__PURE__ */ new _optionMjs.None());
function configure_server_component(config) {
    return new _runtimeMjs.Config(config.open_shadow_root, config.adopt_styles, _dictMjs.from_list(_listMjs.reverse(config.attributes)), _dictMjs.from_list(_listMjs.reverse(config.properties)), _dictMjs.from_list(_listMjs.reverse(config.contexts)), config.on_connect, config.on_disconnect);
}
function configure(options) {
    return _listMjs.fold(options, default_config, (config, option)=>{
        return option.apply(config);
    });
}

},{"../../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../../gleam_stdlib/gleam/dict.mjs":"b8yrU","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../gleam.mjs":"jNPQG","../../lustre/effect.mjs":"iAEPi","../../lustre/internals/constants.mjs":"gKFR6","../../lustre/runtime/server/runtime.mjs":"8rUwG","../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"8rUwG":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "State", ()=>State);
parcelHelpers.export(exports, "State$State", ()=>State$State);
parcelHelpers.export(exports, "State$isState", ()=>State$isState);
parcelHelpers.export(exports, "State$State$self", ()=>State$State$self);
parcelHelpers.export(exports, "State$State$0", ()=>State$State$0);
parcelHelpers.export(exports, "State$State$selector", ()=>State$State$selector);
parcelHelpers.export(exports, "State$State$1", ()=>State$State$1);
parcelHelpers.export(exports, "State$State$base_selector", ()=>State$State$base_selector);
parcelHelpers.export(exports, "State$State$2", ()=>State$State$2);
parcelHelpers.export(exports, "State$State$model", ()=>State$State$model);
parcelHelpers.export(exports, "State$State$3", ()=>State$State$3);
parcelHelpers.export(exports, "State$State$update", ()=>State$State$update);
parcelHelpers.export(exports, "State$State$4", ()=>State$State$4);
parcelHelpers.export(exports, "State$State$view", ()=>State$State$view);
parcelHelpers.export(exports, "State$State$5", ()=>State$State$5);
parcelHelpers.export(exports, "State$State$config", ()=>State$State$config);
parcelHelpers.export(exports, "State$State$6", ()=>State$State$6);
parcelHelpers.export(exports, "State$State$vdom", ()=>State$State$vdom);
parcelHelpers.export(exports, "State$State$7", ()=>State$State$7);
parcelHelpers.export(exports, "State$State$cache", ()=>State$State$cache);
parcelHelpers.export(exports, "State$State$8", ()=>State$State$8);
parcelHelpers.export(exports, "State$State$providers", ()=>State$State$providers);
parcelHelpers.export(exports, "State$State$9", ()=>State$State$9);
parcelHelpers.export(exports, "State$State$subscribers", ()=>State$State$subscribers);
parcelHelpers.export(exports, "State$State$10", ()=>State$State$10);
parcelHelpers.export(exports, "State$State$callbacks", ()=>State$State$callbacks);
parcelHelpers.export(exports, "State$State$11", ()=>State$State$11);
parcelHelpers.export(exports, "Config", ()=>Config);
parcelHelpers.export(exports, "Config$Config", ()=>Config$Config);
parcelHelpers.export(exports, "Config$isConfig", ()=>Config$isConfig);
parcelHelpers.export(exports, "Config$Config$open_shadow_root", ()=>Config$Config$open_shadow_root);
parcelHelpers.export(exports, "Config$Config$0", ()=>Config$Config$0);
parcelHelpers.export(exports, "Config$Config$adopt_styles", ()=>Config$Config$adopt_styles);
parcelHelpers.export(exports, "Config$Config$1", ()=>Config$Config$1);
parcelHelpers.export(exports, "Config$Config$attributes", ()=>Config$Config$attributes);
parcelHelpers.export(exports, "Config$Config$2", ()=>Config$Config$2);
parcelHelpers.export(exports, "Config$Config$properties", ()=>Config$Config$properties);
parcelHelpers.export(exports, "Config$Config$3", ()=>Config$Config$3);
parcelHelpers.export(exports, "Config$Config$contexts", ()=>Config$Config$contexts);
parcelHelpers.export(exports, "Config$Config$4", ()=>Config$Config$4);
parcelHelpers.export(exports, "Config$Config$on_connect", ()=>Config$Config$on_connect);
parcelHelpers.export(exports, "Config$Config$5", ()=>Config$Config$5);
parcelHelpers.export(exports, "Config$Config$on_disconnect", ()=>Config$Config$on_disconnect);
parcelHelpers.export(exports, "Config$Config$6", ()=>Config$Config$6);
parcelHelpers.export(exports, "ClientDispatchedMessage", ()=>ClientDispatchedMessage);
parcelHelpers.export(exports, "Message$ClientDispatchedMessage", ()=>Message$ClientDispatchedMessage);
parcelHelpers.export(exports, "Message$isClientDispatchedMessage", ()=>Message$isClientDispatchedMessage);
parcelHelpers.export(exports, "Message$ClientDispatchedMessage$message", ()=>Message$ClientDispatchedMessage$message);
parcelHelpers.export(exports, "Message$ClientDispatchedMessage$0", ()=>Message$ClientDispatchedMessage$0);
parcelHelpers.export(exports, "ClientRegisteredSubject", ()=>ClientRegisteredSubject);
parcelHelpers.export(exports, "Message$ClientRegisteredSubject", ()=>Message$ClientRegisteredSubject);
parcelHelpers.export(exports, "Message$isClientRegisteredSubject", ()=>Message$isClientRegisteredSubject);
parcelHelpers.export(exports, "Message$ClientRegisteredSubject$client", ()=>Message$ClientRegisteredSubject$client);
parcelHelpers.export(exports, "Message$ClientRegisteredSubject$0", ()=>Message$ClientRegisteredSubject$0);
parcelHelpers.export(exports, "ClientDeregisteredSubject", ()=>ClientDeregisteredSubject);
parcelHelpers.export(exports, "Message$ClientDeregisteredSubject", ()=>Message$ClientDeregisteredSubject);
parcelHelpers.export(exports, "Message$isClientDeregisteredSubject", ()=>Message$isClientDeregisteredSubject);
parcelHelpers.export(exports, "Message$ClientDeregisteredSubject$client", ()=>Message$ClientDeregisteredSubject$client);
parcelHelpers.export(exports, "Message$ClientDeregisteredSubject$0", ()=>Message$ClientDeregisteredSubject$0);
parcelHelpers.export(exports, "ClientRegisteredCallback", ()=>ClientRegisteredCallback);
parcelHelpers.export(exports, "Message$ClientRegisteredCallback", ()=>Message$ClientRegisteredCallback);
parcelHelpers.export(exports, "Message$isClientRegisteredCallback", ()=>Message$isClientRegisteredCallback);
parcelHelpers.export(exports, "Message$ClientRegisteredCallback$callback", ()=>Message$ClientRegisteredCallback$callback);
parcelHelpers.export(exports, "Message$ClientRegisteredCallback$0", ()=>Message$ClientRegisteredCallback$0);
parcelHelpers.export(exports, "ClientDeregisteredCallback", ()=>ClientDeregisteredCallback);
parcelHelpers.export(exports, "Message$ClientDeregisteredCallback", ()=>Message$ClientDeregisteredCallback);
parcelHelpers.export(exports, "Message$isClientDeregisteredCallback", ()=>Message$isClientDeregisteredCallback);
parcelHelpers.export(exports, "Message$ClientDeregisteredCallback$callback", ()=>Message$ClientDeregisteredCallback$callback);
parcelHelpers.export(exports, "Message$ClientDeregisteredCallback$0", ()=>Message$ClientDeregisteredCallback$0);
parcelHelpers.export(exports, "EffectAddedSelector", ()=>EffectAddedSelector);
parcelHelpers.export(exports, "Message$EffectAddedSelector", ()=>Message$EffectAddedSelector);
parcelHelpers.export(exports, "Message$isEffectAddedSelector", ()=>Message$isEffectAddedSelector);
parcelHelpers.export(exports, "Message$EffectAddedSelector$selector", ()=>Message$EffectAddedSelector$selector);
parcelHelpers.export(exports, "Message$EffectAddedSelector$0", ()=>Message$EffectAddedSelector$0);
parcelHelpers.export(exports, "EffectDispatchedMessage", ()=>EffectDispatchedMessage);
parcelHelpers.export(exports, "Message$EffectDispatchedMessage", ()=>Message$EffectDispatchedMessage);
parcelHelpers.export(exports, "Message$isEffectDispatchedMessage", ()=>Message$isEffectDispatchedMessage);
parcelHelpers.export(exports, "Message$EffectDispatchedMessage$message", ()=>Message$EffectDispatchedMessage$message);
parcelHelpers.export(exports, "Message$EffectDispatchedMessage$0", ()=>Message$EffectDispatchedMessage$0);
parcelHelpers.export(exports, "EffectEmitEvent", ()=>EffectEmitEvent);
parcelHelpers.export(exports, "Message$EffectEmitEvent", ()=>Message$EffectEmitEvent);
parcelHelpers.export(exports, "Message$isEffectEmitEvent", ()=>Message$isEffectEmitEvent);
parcelHelpers.export(exports, "Message$EffectEmitEvent$name", ()=>Message$EffectEmitEvent$name);
parcelHelpers.export(exports, "Message$EffectEmitEvent$0", ()=>Message$EffectEmitEvent$0);
parcelHelpers.export(exports, "Message$EffectEmitEvent$data", ()=>Message$EffectEmitEvent$data);
parcelHelpers.export(exports, "Message$EffectEmitEvent$1", ()=>Message$EffectEmitEvent$1);
parcelHelpers.export(exports, "EffectProvidedValue", ()=>EffectProvidedValue);
parcelHelpers.export(exports, "Message$EffectProvidedValue", ()=>Message$EffectProvidedValue);
parcelHelpers.export(exports, "Message$isEffectProvidedValue", ()=>Message$isEffectProvidedValue);
parcelHelpers.export(exports, "Message$EffectProvidedValue$key", ()=>Message$EffectProvidedValue$key);
parcelHelpers.export(exports, "Message$EffectProvidedValue$0", ()=>Message$EffectProvidedValue$0);
parcelHelpers.export(exports, "Message$EffectProvidedValue$value", ()=>Message$EffectProvidedValue$value);
parcelHelpers.export(exports, "Message$EffectProvidedValue$1", ()=>Message$EffectProvidedValue$1);
parcelHelpers.export(exports, "MonitorReportedDown", ()=>MonitorReportedDown);
parcelHelpers.export(exports, "Message$MonitorReportedDown", ()=>Message$MonitorReportedDown);
parcelHelpers.export(exports, "Message$isMonitorReportedDown", ()=>Message$isMonitorReportedDown);
parcelHelpers.export(exports, "Message$MonitorReportedDown$monitor", ()=>Message$MonitorReportedDown$monitor);
parcelHelpers.export(exports, "Message$MonitorReportedDown$0", ()=>Message$MonitorReportedDown$0);
parcelHelpers.export(exports, "SystemRequestedShutdown", ()=>SystemRequestedShutdown);
parcelHelpers.export(exports, "Message$SystemRequestedShutdown", ()=>Message$SystemRequestedShutdown);
parcelHelpers.export(exports, "Message$isSystemRequestedShutdown", ()=>Message$isSystemRequestedShutdown);
parcelHelpers.export(exports, "start", ()=>start);
var _processMjs = require("../../../../gleam_erlang/gleam/erlang/process.mjs");
var _jsonMjs = require("../../../../gleam_json/gleam/json.mjs");
var _actorMjs = require("../../../../gleam_otp/gleam/otp/actor.mjs");
var _dictMjs = require("../../../../gleam_stdlib/gleam/dict.mjs");
var _decodeMjs = require("../../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _optionMjs = require("../../../../gleam_stdlib/gleam/option.mjs");
var _setMjs = require("../../../../gleam_stdlib/gleam/set.mjs");
var _gleamMjs = require("../../../gleam.mjs");
var _effectMjs = require("../../../lustre/effect.mjs");
var _transportMjs = require("../../../lustre/runtime/transport.mjs");
var _cacheMjs = require("../../../lustre/vdom/cache.mjs");
var _vnodeMjs = require("../../../lustre/vdom/vnode.mjs");
class State extends (0, _gleamMjs.CustomType) {
    constructor(self, selector, base_selector, model, update, view, config, vdom, cache, providers, subscribers, callbacks){
        super();
        this.self = self;
        this.selector = selector;
        this.base_selector = base_selector;
        this.model = model;
        this.update = update;
        this.view = view;
        this.config = config;
        this.vdom = vdom;
        this.cache = cache;
        this.providers = providers;
        this.subscribers = subscribers;
        this.callbacks = callbacks;
    }
}
const State$State = (self, selector, base_selector, model, update, view, config, vdom, cache, providers, subscribers, callbacks)=>new State(self, selector, base_selector, model, update, view, config, vdom, cache, providers, subscribers, callbacks);
const State$isState = (value)=>value instanceof State;
const State$State$self = (value)=>value.self;
const State$State$0 = (value)=>value.self;
const State$State$selector = (value)=>value.selector;
const State$State$1 = (value)=>value.selector;
const State$State$base_selector = (value)=>value.base_selector;
const State$State$2 = (value)=>value.base_selector;
const State$State$model = (value)=>value.model;
const State$State$3 = (value)=>value.model;
const State$State$update = (value)=>value.update;
const State$State$4 = (value)=>value.update;
const State$State$view = (value)=>value.view;
const State$State$5 = (value)=>value.view;
const State$State$config = (value)=>value.config;
const State$State$6 = (value)=>value.config;
const State$State$vdom = (value)=>value.vdom;
const State$State$7 = (value)=>value.vdom;
const State$State$cache = (value)=>value.cache;
const State$State$8 = (value)=>value.cache;
const State$State$providers = (value)=>value.providers;
const State$State$9 = (value)=>value.providers;
const State$State$subscribers = (value)=>value.subscribers;
const State$State$10 = (value)=>value.subscribers;
const State$State$callbacks = (value)=>value.callbacks;
const State$State$11 = (value)=>value.callbacks;
class Config extends (0, _gleamMjs.CustomType) {
    constructor(open_shadow_root, adopt_styles, attributes, properties, contexts, on_connect, on_disconnect){
        super();
        this.open_shadow_root = open_shadow_root;
        this.adopt_styles = adopt_styles;
        this.attributes = attributes;
        this.properties = properties;
        this.contexts = contexts;
        this.on_connect = on_connect;
        this.on_disconnect = on_disconnect;
    }
}
const Config$Config = (open_shadow_root, adopt_styles, attributes, properties, contexts, on_connect, on_disconnect)=>new Config(open_shadow_root, adopt_styles, attributes, properties, contexts, on_connect, on_disconnect);
const Config$isConfig = (value)=>value instanceof Config;
const Config$Config$open_shadow_root = (value)=>value.open_shadow_root;
const Config$Config$0 = (value)=>value.open_shadow_root;
const Config$Config$adopt_styles = (value)=>value.adopt_styles;
const Config$Config$1 = (value)=>value.adopt_styles;
const Config$Config$attributes = (value)=>value.attributes;
const Config$Config$2 = (value)=>value.attributes;
const Config$Config$properties = (value)=>value.properties;
const Config$Config$3 = (value)=>value.properties;
const Config$Config$contexts = (value)=>value.contexts;
const Config$Config$4 = (value)=>value.contexts;
const Config$Config$on_connect = (value)=>value.on_connect;
const Config$Config$5 = (value)=>value.on_connect;
const Config$Config$on_disconnect = (value)=>value.on_disconnect;
const Config$Config$6 = (value)=>value.on_disconnect;
class ClientDispatchedMessage extends (0, _gleamMjs.CustomType) {
    constructor(message){
        super();
        this.message = message;
    }
}
const Message$ClientDispatchedMessage = (message)=>new ClientDispatchedMessage(message);
const Message$isClientDispatchedMessage = (value)=>value instanceof ClientDispatchedMessage;
const Message$ClientDispatchedMessage$message = (value)=>value.message;
const Message$ClientDispatchedMessage$0 = (value)=>value.message;
class ClientRegisteredSubject extends (0, _gleamMjs.CustomType) {
    constructor(client){
        super();
        this.client = client;
    }
}
const Message$ClientRegisteredSubject = (client)=>new ClientRegisteredSubject(client);
const Message$isClientRegisteredSubject = (value)=>value instanceof ClientRegisteredSubject;
const Message$ClientRegisteredSubject$client = (value)=>value.client;
const Message$ClientRegisteredSubject$0 = (value)=>value.client;
class ClientDeregisteredSubject extends (0, _gleamMjs.CustomType) {
    constructor(client){
        super();
        this.client = client;
    }
}
const Message$ClientDeregisteredSubject = (client)=>new ClientDeregisteredSubject(client);
const Message$isClientDeregisteredSubject = (value)=>value instanceof ClientDeregisteredSubject;
const Message$ClientDeregisteredSubject$client = (value)=>value.client;
const Message$ClientDeregisteredSubject$0 = (value)=>value.client;
class ClientRegisteredCallback extends (0, _gleamMjs.CustomType) {
    constructor(callback){
        super();
        this.callback = callback;
    }
}
const Message$ClientRegisteredCallback = (callback)=>new ClientRegisteredCallback(callback);
const Message$isClientRegisteredCallback = (value)=>value instanceof ClientRegisteredCallback;
const Message$ClientRegisteredCallback$callback = (value)=>value.callback;
const Message$ClientRegisteredCallback$0 = (value)=>value.callback;
class ClientDeregisteredCallback extends (0, _gleamMjs.CustomType) {
    constructor(callback){
        super();
        this.callback = callback;
    }
}
const Message$ClientDeregisteredCallback = (callback)=>new ClientDeregisteredCallback(callback);
const Message$isClientDeregisteredCallback = (value)=>value instanceof ClientDeregisteredCallback;
const Message$ClientDeregisteredCallback$callback = (value)=>value.callback;
const Message$ClientDeregisteredCallback$0 = (value)=>value.callback;
class EffectAddedSelector extends (0, _gleamMjs.CustomType) {
    constructor(selector){
        super();
        this.selector = selector;
    }
}
const Message$EffectAddedSelector = (selector)=>new EffectAddedSelector(selector);
const Message$isEffectAddedSelector = (value)=>value instanceof EffectAddedSelector;
const Message$EffectAddedSelector$selector = (value)=>value.selector;
const Message$EffectAddedSelector$0 = (value)=>value.selector;
class EffectDispatchedMessage extends (0, _gleamMjs.CustomType) {
    constructor(message){
        super();
        this.message = message;
    }
}
const Message$EffectDispatchedMessage = (message)=>new EffectDispatchedMessage(message);
const Message$isEffectDispatchedMessage = (value)=>value instanceof EffectDispatchedMessage;
const Message$EffectDispatchedMessage$message = (value)=>value.message;
const Message$EffectDispatchedMessage$0 = (value)=>value.message;
class EffectEmitEvent extends (0, _gleamMjs.CustomType) {
    constructor(name, data){
        super();
        this.name = name;
        this.data = data;
    }
}
const Message$EffectEmitEvent = (name, data)=>new EffectEmitEvent(name, data);
const Message$isEffectEmitEvent = (value)=>value instanceof EffectEmitEvent;
const Message$EffectEmitEvent$name = (value)=>value.name;
const Message$EffectEmitEvent$0 = (value)=>value.name;
const Message$EffectEmitEvent$data = (value)=>value.data;
const Message$EffectEmitEvent$1 = (value)=>value.data;
class EffectProvidedValue extends (0, _gleamMjs.CustomType) {
    constructor(key, value){
        super();
        this.key = key;
        this.value = value;
    }
}
const Message$EffectProvidedValue = (key, value)=>new EffectProvidedValue(key, value);
const Message$isEffectProvidedValue = (value)=>value instanceof EffectProvidedValue;
const Message$EffectProvidedValue$key = (value)=>value.key;
const Message$EffectProvidedValue$0 = (value)=>value.key;
const Message$EffectProvidedValue$value = (value)=>value.value;
const Message$EffectProvidedValue$1 = (value)=>value.value;
class MonitorReportedDown extends (0, _gleamMjs.CustomType) {
    constructor(monitor){
        super();
        this.monitor = monitor;
    }
}
const Message$MonitorReportedDown = (monitor)=>new MonitorReportedDown(monitor);
const Message$isMonitorReportedDown = (value)=>value instanceof MonitorReportedDown;
const Message$MonitorReportedDown$monitor = (value)=>value.monitor;
const Message$MonitorReportedDown$0 = (value)=>value.monitor;
class SystemRequestedShutdown extends (0, _gleamMjs.CustomType) {
}
const Message$SystemRequestedShutdown = ()=>new SystemRequestedShutdown();
const Message$isSystemRequestedShutdown = (value)=>value instanceof SystemRequestedShutdown;
function start(_, _1, _2, _3, _4, _5) {
    return new (0, _gleamMjs.Error)(new _actorMjs.InitFailed("Not Erlang"));
}

},{"../../../../gleam_erlang/gleam/erlang/process.mjs":"jb30g","../../../../gleam_json/gleam/json.mjs":"8Pq32","../../../../gleam_otp/gleam/otp/actor.mjs":"jWzax","../../../../gleam_stdlib/gleam/dict.mjs":"b8yrU","../../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../../../gleam_stdlib/gleam/set.mjs":"5SoAd","../../../gleam.mjs":"jNPQG","../../../lustre/effect.mjs":"iAEPi","../../../lustre/runtime/transport.mjs":"9jG6q","../../../lustre/vdom/cache.mjs":"aEh50","../../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5SoAd":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Creates a new empty set.
 */ parcelHelpers.export(exports, "new$", ()=>new$);
/**
 * Gets the number of members in a set.
 *
 * This function runs in constant time.
 *
 * ## Examples
 *
 * ```gleam
 * assert new()
 *   |> insert(1)
 *   |> insert(2)
 *   |> size
 *   == 2
 * ```
 */ parcelHelpers.export(exports, "size", ()=>size);
/**
 * Determines whether or not the set is empty.
 *
 * ## Examples
 *
 * ```gleam
 * assert new() |> is_empty
 * ```
 *
 * ```gleam
 * assert !{ new() |> insert(1) |> is_empty }
 * ```
 */ parcelHelpers.export(exports, "is_empty", ()=>is_empty);
/**
 * Checks whether a set contains a given member.
 *
 * This function runs in logarithmic time.
 *
 * ## Examples
 *
 * ```gleam
 * assert new()
 *   |> insert(2)
 *   |> contains(2)
 * ```
 *
 * ```gleam
 * assert !{
 *   new()
 *   |> insert(2)
 *   |> contains(1)
 * }
 * ```
 */ parcelHelpers.export(exports, "contains", ()=>contains);
/**
 * Removes a member from a set. If the set does not contain the member then
 * the set is returned unchanged.
 *
 * This function runs in logarithmic time.
 *
 * ## Examples
 *
 * ```gleam
 * assert !{
 *   new()
 *   |> insert(2)
 *   |> delete(2)
 *   |> contains(2)
 * }
 * ```
 */ parcelHelpers.export(exports, "delete$", ()=>delete$);
/**
 * Converts the set into a list of the contained members.
 *
 * The list has no specific ordering, any unintentional ordering may change in
 * future versions of Gleam or Erlang.
 *
 * This function runs in linear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert new() |> insert(2) |> to_list == [2]
 * ```
 */ parcelHelpers.export(exports, "to_list", ()=>to_list);
/**
 * Combines all entries into a single value by calling a given function on each
 * one.
 *
 * Sets are not ordered so the values are not returned in any specific order.
 * Do not write code that relies on the order entries are used by this
 * function as it may change in later versions of Gleam or Erlang.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([1, 3, 9])
 *   |> fold(0, fn(accumulator, member) { accumulator + member })
 *   == 13
 * ```
 */ parcelHelpers.export(exports, "fold", ()=>fold);
/**
 * Creates a new set from an existing set, minus any members that a given
 * function returns `False` for.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 *
 * assert from_list([1, 4, 6, 3, 675, 44, 67])
 *   |> filter(keeping: int.is_even)
 *   |> to_list
 *   == [4, 6, 44]
 * ```
 */ parcelHelpers.export(exports, "filter", ()=>filter);
/**
 * Creates a new set from a given set with all the same entries except any
 * entry found on the given list.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([1, 2, 3, 4])
 *   |> drop([1, 3])
 *   |> to_list
 *   == [2, 4]
 * ```
 */ parcelHelpers.export(exports, "drop", ()=>drop);
/**
 * Creates a new set from a given set, only including any members which are in
 * a given list.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([1, 2, 3])
 *   |> take([1, 3, 5])
 *   |> to_list
 *   == [1, 3]
 * ```
 */ parcelHelpers.export(exports, "take", ()=>take);
/**
 * Creates a new set that contains members that are present in both given sets.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert intersection(from_list([1, 2]), from_list([2, 3])) |> to_list
 *   == [2]
 * ```
 */ parcelHelpers.export(exports, "intersection", ()=>intersection);
/**
 * Creates a new set that contains members that are present in the first set
 * but not the second.
 *
 * ## Examples
 *
 * ```gleam
 * assert difference(from_list([1, 2]), from_list([2, 3, 4])) |> to_list
 *   == [1]
 * ```
 */ parcelHelpers.export(exports, "difference", ()=>difference);
/**
 * Determines if a set is fully contained by another.
 *
 * ## Examples
 *
 * ```gleam
 * assert is_subset(from_list([1]), from_list([1, 2]))
 * ```
 *
 * ```gleam
 * assert !is_subset(from_list([1, 2, 3]), from_list([3, 4, 5]))
 * ```
 */ parcelHelpers.export(exports, "is_subset", ()=>is_subset);
/**
 * Determines if two sets contain no common members
 *
 * ## Examples
 *
 * ```gleam
 * assert is_disjoint(from_list([1, 2, 3]), from_list([4, 5, 6]))
 * ```
 *
 * ```gleam
 * assert !is_disjoint(from_list([1, 2, 3]), from_list([3, 4, 5]))
 * ```
 */ parcelHelpers.export(exports, "is_disjoint", ()=>is_disjoint);
/**
 * Calls a function for each member in a set, discarding the return
 * value.
 *
 * Useful for producing a side effect for every item of a set.
 *
 * The order of elements in the iteration is an implementation detail that
 * should not be relied upon.
 *
 * ## Examples
 *
 * ```gleam
 * let set = from_list(["apple", "banana", "cherry"])
 *
 * assert each(set, io.println) == Nil
 * // apple
 * // banana
 * // cherry
 * ```
 */ parcelHelpers.export(exports, "each", ()=>each);
/**
 * Inserts a member into the set.
 *
 * This function runs in logarithmic time.
 *
 * ## Examples
 *
 * ```gleam
 * assert new()
 *   |> insert(1)
 *   |> insert(2)
 *   |> size
 *   == 2
 * ```
 */ parcelHelpers.export(exports, "insert", ()=>insert);
/**
 * Creates a new set of the members in a given list.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * import gleam/int
 * import gleam/list
 *
 * assert [1, 1, 2, 4, 3, 2]
 *   |> from_list
 *   |> to_list
 *   |> list.sort(by: int.compare)
 *   == [1, 2, 3, 4]
 * ```
 */ parcelHelpers.export(exports, "from_list", ()=>from_list);
/**
 * Creates a new set from a given set with the result of applying the given
 * function to each member.
 *
 * ## Examples
 *
 * ```gleam
 * assert from_list([1, 2, 3, 4])
 *   |> map(with: fn(x) { x * 2 })
 *   |> to_list
 *   == [2, 4, 6, 8]
 * ```
 */ parcelHelpers.export(exports, "map", ()=>map);
/**
 * Creates a new set that contains all members of both given sets.
 *
 * This function runs in loglinear time.
 *
 * ## Examples
 *
 * ```gleam
 * assert union(from_list([1, 2]), from_list([2, 3])) |> to_list
 *   == [1, 2, 3]
 * ```
 */ parcelHelpers.export(exports, "union", ()=>union);
/**
 * Creates a new set that contains members that are present in either set, but
 * not both.
 *
 * ## Examples
 *
 * ```gleam
 * assert symmetric_difference(from_list([1, 2, 3]), from_list([3, 4]))
 *   |> to_list
 *   == [1, 2, 4]
 * ```
 */ parcelHelpers.export(exports, "symmetric_difference", ()=>symmetric_difference);
var _gleamMjs = require("../gleam.mjs");
var _dictMjs = require("../gleam/dict.mjs");
var _listMjs = require("../gleam/list.mjs");
var _resultMjs = require("../gleam/result.mjs");
class Set extends (0, _gleamMjs.CustomType) {
    constructor(dict){
        super();
        this.dict = dict;
    }
}
const token = undefined;
function new$() {
    return new Set(_dictMjs.new$());
}
function size(set) {
    return _dictMjs.size(set.dict);
}
function is_empty(set) {
    return (0, _gleamMjs.isEqual)(set, new$());
}
function contains(set, member) {
    let _pipe = set.dict;
    let _pipe$1 = _dictMjs.get(_pipe, member);
    return _resultMjs.is_ok(_pipe$1);
}
function delete$(set, member) {
    return new Set(_dictMjs.delete$(set.dict, member));
}
function to_list(set) {
    return _dictMjs.keys(set.dict);
}
function fold(set, initial, reducer) {
    return _dictMjs.fold(set.dict, initial, (a, k, _)=>{
        return reducer(a, k);
    });
}
function filter(set, predicate) {
    return new Set(_dictMjs.filter(set.dict, (m, _)=>{
        return predicate(m);
    }));
}
function drop(set, disallowed) {
    return _listMjs.fold(disallowed, set, delete$);
}
function take(set, desired) {
    return new Set(_dictMjs.take(set.dict, desired));
}
function order(first, second) {
    let $ = _dictMjs.size(first.dict) > _dictMjs.size(second.dict);
    if ($) return [
        first,
        second
    ];
    else return [
        second,
        first
    ];
}
function intersection(first, second) {
    let $ = order(first, second);
    let larger;
    let smaller;
    larger = $[0];
    smaller = $[1];
    return take(larger, to_list(smaller));
}
function difference(first, second) {
    return drop(first, to_list(second));
}
function is_subset(first, second) {
    return (0, _gleamMjs.isEqual)(intersection(first, second), first);
}
function is_disjoint(first, second) {
    return (0, _gleamMjs.isEqual)(intersection(first, second), new$());
}
function each(set, fun) {
    return fold(set, undefined, (nil, member)=>{
        fun(member);
        return nil;
    });
}
function insert(set, member) {
    return new Set(_dictMjs.insert(set.dict, member, token));
}
function from_list(members) {
    let dict = _listMjs.fold(members, _dictMjs.new$(), (m, k)=>{
        return _dictMjs.insert(m, k, token);
    });
    return new Set(dict);
}
function map(set, fun) {
    return fold(set, new$(), (acc, member)=>{
        return insert(acc, fun(member));
    });
}
function union(first, second) {
    let $ = order(first, second);
    let larger;
    let smaller;
    larger = $[0];
    smaller = $[1];
    return fold(smaller, larger, insert);
}
function symmetric_difference(first, second) {
    return difference(union(first, second), intersection(first, second));
}

},{"../gleam.mjs":"aiPrb","../gleam/dict.mjs":"b8yrU","../gleam/list.mjs":"8dUwY","../gleam/result.mjs":"oBmFG","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"9jG6q":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Mount", ()=>Mount);
parcelHelpers.export(exports, "ClientMessage$Mount", ()=>ClientMessage$Mount);
parcelHelpers.export(exports, "ClientMessage$isMount", ()=>ClientMessage$isMount);
parcelHelpers.export(exports, "ClientMessage$Mount$kind", ()=>ClientMessage$Mount$kind);
parcelHelpers.export(exports, "ClientMessage$Mount$0", ()=>ClientMessage$Mount$0);
parcelHelpers.export(exports, "ClientMessage$Mount$open_shadow_root", ()=>ClientMessage$Mount$open_shadow_root);
parcelHelpers.export(exports, "ClientMessage$Mount$1", ()=>ClientMessage$Mount$1);
parcelHelpers.export(exports, "ClientMessage$Mount$will_adopt_styles", ()=>ClientMessage$Mount$will_adopt_styles);
parcelHelpers.export(exports, "ClientMessage$Mount$2", ()=>ClientMessage$Mount$2);
parcelHelpers.export(exports, "ClientMessage$Mount$observed_attributes", ()=>ClientMessage$Mount$observed_attributes);
parcelHelpers.export(exports, "ClientMessage$Mount$3", ()=>ClientMessage$Mount$3);
parcelHelpers.export(exports, "ClientMessage$Mount$observed_properties", ()=>ClientMessage$Mount$observed_properties);
parcelHelpers.export(exports, "ClientMessage$Mount$4", ()=>ClientMessage$Mount$4);
parcelHelpers.export(exports, "ClientMessage$Mount$requested_contexts", ()=>ClientMessage$Mount$requested_contexts);
parcelHelpers.export(exports, "ClientMessage$Mount$5", ()=>ClientMessage$Mount$5);
parcelHelpers.export(exports, "ClientMessage$Mount$provided_contexts", ()=>ClientMessage$Mount$provided_contexts);
parcelHelpers.export(exports, "ClientMessage$Mount$6", ()=>ClientMessage$Mount$6);
parcelHelpers.export(exports, "ClientMessage$Mount$vdom", ()=>ClientMessage$Mount$vdom);
parcelHelpers.export(exports, "ClientMessage$Mount$7", ()=>ClientMessage$Mount$7);
parcelHelpers.export(exports, "ClientMessage$Mount$memos", ()=>ClientMessage$Mount$memos);
parcelHelpers.export(exports, "ClientMessage$Mount$8", ()=>ClientMessage$Mount$8);
parcelHelpers.export(exports, "Reconcile", ()=>Reconcile);
parcelHelpers.export(exports, "ClientMessage$Reconcile", ()=>ClientMessage$Reconcile);
parcelHelpers.export(exports, "ClientMessage$isReconcile", ()=>ClientMessage$isReconcile);
parcelHelpers.export(exports, "ClientMessage$Reconcile$kind", ()=>ClientMessage$Reconcile$kind);
parcelHelpers.export(exports, "ClientMessage$Reconcile$0", ()=>ClientMessage$Reconcile$0);
parcelHelpers.export(exports, "ClientMessage$Reconcile$patch", ()=>ClientMessage$Reconcile$patch);
parcelHelpers.export(exports, "ClientMessage$Reconcile$1", ()=>ClientMessage$Reconcile$1);
parcelHelpers.export(exports, "ClientMessage$Reconcile$memos", ()=>ClientMessage$Reconcile$memos);
parcelHelpers.export(exports, "ClientMessage$Reconcile$2", ()=>ClientMessage$Reconcile$2);
parcelHelpers.export(exports, "Emit", ()=>Emit);
parcelHelpers.export(exports, "ClientMessage$Emit", ()=>ClientMessage$Emit);
parcelHelpers.export(exports, "ClientMessage$isEmit", ()=>ClientMessage$isEmit);
parcelHelpers.export(exports, "ClientMessage$Emit$kind", ()=>ClientMessage$Emit$kind);
parcelHelpers.export(exports, "ClientMessage$Emit$0", ()=>ClientMessage$Emit$0);
parcelHelpers.export(exports, "ClientMessage$Emit$name", ()=>ClientMessage$Emit$name);
parcelHelpers.export(exports, "ClientMessage$Emit$1", ()=>ClientMessage$Emit$1);
parcelHelpers.export(exports, "ClientMessage$Emit$data", ()=>ClientMessage$Emit$data);
parcelHelpers.export(exports, "ClientMessage$Emit$2", ()=>ClientMessage$Emit$2);
parcelHelpers.export(exports, "Provide", ()=>Provide);
parcelHelpers.export(exports, "ClientMessage$Provide", ()=>ClientMessage$Provide);
parcelHelpers.export(exports, "ClientMessage$isProvide", ()=>ClientMessage$isProvide);
parcelHelpers.export(exports, "ClientMessage$Provide$kind", ()=>ClientMessage$Provide$kind);
parcelHelpers.export(exports, "ClientMessage$Provide$0", ()=>ClientMessage$Provide$0);
parcelHelpers.export(exports, "ClientMessage$Provide$key", ()=>ClientMessage$Provide$key);
parcelHelpers.export(exports, "ClientMessage$Provide$1", ()=>ClientMessage$Provide$1);
parcelHelpers.export(exports, "ClientMessage$Provide$value", ()=>ClientMessage$Provide$value);
parcelHelpers.export(exports, "ClientMessage$Provide$2", ()=>ClientMessage$Provide$2);
parcelHelpers.export(exports, "ClientMessage$kind", ()=>ClientMessage$kind);
parcelHelpers.export(exports, "Batch", ()=>Batch);
parcelHelpers.export(exports, "ServerMessage$Batch", ()=>ServerMessage$Batch);
parcelHelpers.export(exports, "ServerMessage$isBatch", ()=>ServerMessage$isBatch);
parcelHelpers.export(exports, "ServerMessage$Batch$kind", ()=>ServerMessage$Batch$kind);
parcelHelpers.export(exports, "ServerMessage$Batch$0", ()=>ServerMessage$Batch$0);
parcelHelpers.export(exports, "ServerMessage$Batch$messages", ()=>ServerMessage$Batch$messages);
parcelHelpers.export(exports, "ServerMessage$Batch$1", ()=>ServerMessage$Batch$1);
parcelHelpers.export(exports, "AttributeChanged", ()=>AttributeChanged);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged", ()=>ServerMessage$AttributeChanged);
parcelHelpers.export(exports, "ServerMessage$isAttributeChanged", ()=>ServerMessage$isAttributeChanged);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$kind", ()=>ServerMessage$AttributeChanged$kind);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$0", ()=>ServerMessage$AttributeChanged$0);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$name", ()=>ServerMessage$AttributeChanged$name);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$1", ()=>ServerMessage$AttributeChanged$1);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$value", ()=>ServerMessage$AttributeChanged$value);
parcelHelpers.export(exports, "ServerMessage$AttributeChanged$2", ()=>ServerMessage$AttributeChanged$2);
parcelHelpers.export(exports, "PropertyChanged", ()=>PropertyChanged);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged", ()=>ServerMessage$PropertyChanged);
parcelHelpers.export(exports, "ServerMessage$isPropertyChanged", ()=>ServerMessage$isPropertyChanged);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$kind", ()=>ServerMessage$PropertyChanged$kind);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$0", ()=>ServerMessage$PropertyChanged$0);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$name", ()=>ServerMessage$PropertyChanged$name);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$1", ()=>ServerMessage$PropertyChanged$1);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$value", ()=>ServerMessage$PropertyChanged$value);
parcelHelpers.export(exports, "ServerMessage$PropertyChanged$2", ()=>ServerMessage$PropertyChanged$2);
parcelHelpers.export(exports, "EventFired", ()=>EventFired);
parcelHelpers.export(exports, "ServerMessage$EventFired", ()=>ServerMessage$EventFired);
parcelHelpers.export(exports, "ServerMessage$isEventFired", ()=>ServerMessage$isEventFired);
parcelHelpers.export(exports, "ServerMessage$EventFired$kind", ()=>ServerMessage$EventFired$kind);
parcelHelpers.export(exports, "ServerMessage$EventFired$0", ()=>ServerMessage$EventFired$0);
parcelHelpers.export(exports, "ServerMessage$EventFired$path", ()=>ServerMessage$EventFired$path);
parcelHelpers.export(exports, "ServerMessage$EventFired$1", ()=>ServerMessage$EventFired$1);
parcelHelpers.export(exports, "ServerMessage$EventFired$name", ()=>ServerMessage$EventFired$name);
parcelHelpers.export(exports, "ServerMessage$EventFired$2", ()=>ServerMessage$EventFired$2);
parcelHelpers.export(exports, "ServerMessage$EventFired$event", ()=>ServerMessage$EventFired$event);
parcelHelpers.export(exports, "ServerMessage$EventFired$3", ()=>ServerMessage$EventFired$3);
parcelHelpers.export(exports, "ContextProvided", ()=>ContextProvided);
parcelHelpers.export(exports, "ServerMessage$ContextProvided", ()=>ServerMessage$ContextProvided);
parcelHelpers.export(exports, "ServerMessage$isContextProvided", ()=>ServerMessage$isContextProvided);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$kind", ()=>ServerMessage$ContextProvided$kind);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$0", ()=>ServerMessage$ContextProvided$0);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$key", ()=>ServerMessage$ContextProvided$key);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$1", ()=>ServerMessage$ContextProvided$1);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$value", ()=>ServerMessage$ContextProvided$value);
parcelHelpers.export(exports, "ServerMessage$ContextProvided$2", ()=>ServerMessage$ContextProvided$2);
parcelHelpers.export(exports, "ServerMessage$kind", ()=>ServerMessage$kind);
parcelHelpers.export(exports, "mount_kind", ()=>mount_kind);
parcelHelpers.export(exports, "reconcile_kind", ()=>reconcile_kind);
parcelHelpers.export(exports, "emit_kind", ()=>emit_kind);
parcelHelpers.export(exports, "provide_kind", ()=>provide_kind);
parcelHelpers.export(exports, "attribute_changed_kind", ()=>attribute_changed_kind);
parcelHelpers.export(exports, "event_fired_kind", ()=>event_fired_kind);
parcelHelpers.export(exports, "property_changed_kind", ()=>property_changed_kind);
parcelHelpers.export(exports, "batch_kind", ()=>batch_kind);
parcelHelpers.export(exports, "context_provided_kind", ()=>context_provided_kind);
parcelHelpers.export(exports, "client_message_to_json", ()=>client_message_to_json);
parcelHelpers.export(exports, "mount", ()=>mount);
parcelHelpers.export(exports, "reconcile", ()=>reconcile);
parcelHelpers.export(exports, "emit", ()=>emit);
parcelHelpers.export(exports, "provide", ()=>provide);
parcelHelpers.export(exports, "attribute_changed", ()=>attribute_changed);
parcelHelpers.export(exports, "event_fired", ()=>event_fired);
parcelHelpers.export(exports, "property_changed", ()=>property_changed);
parcelHelpers.export(exports, "batch", ()=>batch);
parcelHelpers.export(exports, "context_provided", ()=>context_provided);
parcelHelpers.export(exports, "context_provided_decoder", ()=>context_provided_decoder);
parcelHelpers.export(exports, "server_message_decoder", ()=>server_message_decoder);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _dictMjs = require("../../../gleam_stdlib/gleam/dict.mjs");
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _functionMjs = require("../../../gleam_stdlib/gleam/function.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _patchMjs = require("../../lustre/vdom/patch.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
class Mount extends (0, _gleamMjs.CustomType) {
    constructor(kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos){
        super();
        this.kind = kind;
        this.open_shadow_root = open_shadow_root;
        this.will_adopt_styles = will_adopt_styles;
        this.observed_attributes = observed_attributes;
        this.observed_properties = observed_properties;
        this.requested_contexts = requested_contexts;
        this.provided_contexts = provided_contexts;
        this.vdom = vdom;
        this.memos = memos;
    }
}
const ClientMessage$Mount = (kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos)=>new Mount(kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos);
const ClientMessage$isMount = (value)=>value instanceof Mount;
const ClientMessage$Mount$kind = (value)=>value.kind;
const ClientMessage$Mount$0 = (value)=>value.kind;
const ClientMessage$Mount$open_shadow_root = (value)=>value.open_shadow_root;
const ClientMessage$Mount$1 = (value)=>value.open_shadow_root;
const ClientMessage$Mount$will_adopt_styles = (value)=>value.will_adopt_styles;
const ClientMessage$Mount$2 = (value)=>value.will_adopt_styles;
const ClientMessage$Mount$observed_attributes = (value)=>value.observed_attributes;
const ClientMessage$Mount$3 = (value)=>value.observed_attributes;
const ClientMessage$Mount$observed_properties = (value)=>value.observed_properties;
const ClientMessage$Mount$4 = (value)=>value.observed_properties;
const ClientMessage$Mount$requested_contexts = (value)=>value.requested_contexts;
const ClientMessage$Mount$5 = (value)=>value.requested_contexts;
const ClientMessage$Mount$provided_contexts = (value)=>value.provided_contexts;
const ClientMessage$Mount$6 = (value)=>value.provided_contexts;
const ClientMessage$Mount$vdom = (value)=>value.vdom;
const ClientMessage$Mount$7 = (value)=>value.vdom;
const ClientMessage$Mount$memos = (value)=>value.memos;
const ClientMessage$Mount$8 = (value)=>value.memos;
class Reconcile extends (0, _gleamMjs.CustomType) {
    constructor(kind, patch, memos){
        super();
        this.kind = kind;
        this.patch = patch;
        this.memos = memos;
    }
}
const ClientMessage$Reconcile = (kind, patch, memos)=>new Reconcile(kind, patch, memos);
const ClientMessage$isReconcile = (value)=>value instanceof Reconcile;
const ClientMessage$Reconcile$kind = (value)=>value.kind;
const ClientMessage$Reconcile$0 = (value)=>value.kind;
const ClientMessage$Reconcile$patch = (value)=>value.patch;
const ClientMessage$Reconcile$1 = (value)=>value.patch;
const ClientMessage$Reconcile$memos = (value)=>value.memos;
const ClientMessage$Reconcile$2 = (value)=>value.memos;
class Emit extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, data){
        super();
        this.kind = kind;
        this.name = name;
        this.data = data;
    }
}
const ClientMessage$Emit = (kind, name, data)=>new Emit(kind, name, data);
const ClientMessage$isEmit = (value)=>value instanceof Emit;
const ClientMessage$Emit$kind = (value)=>value.kind;
const ClientMessage$Emit$0 = (value)=>value.kind;
const ClientMessage$Emit$name = (value)=>value.name;
const ClientMessage$Emit$1 = (value)=>value.name;
const ClientMessage$Emit$data = (value)=>value.data;
const ClientMessage$Emit$2 = (value)=>value.data;
class Provide extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, value){
        super();
        this.kind = kind;
        this.key = key;
        this.value = value;
    }
}
const ClientMessage$Provide = (kind, key, value)=>new Provide(kind, key, value);
const ClientMessage$isProvide = (value)=>value instanceof Provide;
const ClientMessage$Provide$kind = (value)=>value.kind;
const ClientMessage$Provide$0 = (value)=>value.kind;
const ClientMessage$Provide$key = (value)=>value.key;
const ClientMessage$Provide$1 = (value)=>value.key;
const ClientMessage$Provide$value = (value)=>value.value;
const ClientMessage$Provide$2 = (value)=>value.value;
const ClientMessage$kind = (value)=>value.kind;
class Batch extends (0, _gleamMjs.CustomType) {
    constructor(kind, messages){
        super();
        this.kind = kind;
        this.messages = messages;
    }
}
const ServerMessage$Batch = (kind, messages)=>new Batch(kind, messages);
const ServerMessage$isBatch = (value)=>value instanceof Batch;
const ServerMessage$Batch$kind = (value)=>value.kind;
const ServerMessage$Batch$0 = (value)=>value.kind;
const ServerMessage$Batch$messages = (value)=>value.messages;
const ServerMessage$Batch$1 = (value)=>value.messages;
class AttributeChanged extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, value){
        super();
        this.kind = kind;
        this.name = name;
        this.value = value;
    }
}
const ServerMessage$AttributeChanged = (kind, name, value)=>new AttributeChanged(kind, name, value);
const ServerMessage$isAttributeChanged = (value)=>value instanceof AttributeChanged;
const ServerMessage$AttributeChanged$kind = (value)=>value.kind;
const ServerMessage$AttributeChanged$0 = (value)=>value.kind;
const ServerMessage$AttributeChanged$name = (value)=>value.name;
const ServerMessage$AttributeChanged$1 = (value)=>value.name;
const ServerMessage$AttributeChanged$value = (value)=>value.value;
const ServerMessage$AttributeChanged$2 = (value)=>value.value;
class PropertyChanged extends (0, _gleamMjs.CustomType) {
    constructor(kind, name, value){
        super();
        this.kind = kind;
        this.name = name;
        this.value = value;
    }
}
const ServerMessage$PropertyChanged = (kind, name, value)=>new PropertyChanged(kind, name, value);
const ServerMessage$isPropertyChanged = (value)=>value instanceof PropertyChanged;
const ServerMessage$PropertyChanged$kind = (value)=>value.kind;
const ServerMessage$PropertyChanged$0 = (value)=>value.kind;
const ServerMessage$PropertyChanged$name = (value)=>value.name;
const ServerMessage$PropertyChanged$1 = (value)=>value.name;
const ServerMessage$PropertyChanged$value = (value)=>value.value;
const ServerMessage$PropertyChanged$2 = (value)=>value.value;
class EventFired extends (0, _gleamMjs.CustomType) {
    constructor(kind, path, name, event){
        super();
        this.kind = kind;
        this.path = path;
        this.name = name;
        this.event = event;
    }
}
const ServerMessage$EventFired = (kind, path, name, event)=>new EventFired(kind, path, name, event);
const ServerMessage$isEventFired = (value)=>value instanceof EventFired;
const ServerMessage$EventFired$kind = (value)=>value.kind;
const ServerMessage$EventFired$0 = (value)=>value.kind;
const ServerMessage$EventFired$path = (value)=>value.path;
const ServerMessage$EventFired$1 = (value)=>value.path;
const ServerMessage$EventFired$name = (value)=>value.name;
const ServerMessage$EventFired$2 = (value)=>value.name;
const ServerMessage$EventFired$event = (value)=>value.event;
const ServerMessage$EventFired$3 = (value)=>value.event;
class ContextProvided extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, value){
        super();
        this.kind = kind;
        this.key = key;
        this.value = value;
    }
}
const ServerMessage$ContextProvided = (kind, key, value)=>new ContextProvided(kind, key, value);
const ServerMessage$isContextProvided = (value)=>value instanceof ContextProvided;
const ServerMessage$ContextProvided$kind = (value)=>value.kind;
const ServerMessage$ContextProvided$0 = (value)=>value.kind;
const ServerMessage$ContextProvided$key = (value)=>value.key;
const ServerMessage$ContextProvided$1 = (value)=>value.key;
const ServerMessage$ContextProvided$value = (value)=>value.value;
const ServerMessage$ContextProvided$2 = (value)=>value.value;
const ServerMessage$kind = (value)=>value.kind;
const mount_kind = 0;
const reconcile_kind = 1;
const emit_kind = 2;
const provide_kind = 3;
const attribute_changed_kind = 0;
const event_fired_kind = 1;
const property_changed_kind = 2;
const batch_kind = 3;
const context_provided_kind = 4;
function mount_to_json(kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos) {
    return _jsonMjs.object((0, _gleamMjs.toList)([
        [
            "kind",
            _jsonMjs.int(kind)
        ],
        [
            "open_shadow_root",
            _jsonMjs.bool(open_shadow_root)
        ],
        [
            "will_adopt_styles",
            _jsonMjs.bool(will_adopt_styles)
        ],
        [
            "observed_attributes",
            _jsonMjs.array(observed_attributes, _jsonMjs.string)
        ],
        [
            "observed_properties",
            _jsonMjs.array(observed_properties, _jsonMjs.string)
        ],
        [
            "requested_contexts",
            _jsonMjs.array(requested_contexts, _jsonMjs.string)
        ],
        [
            "provided_contexts",
            _jsonMjs.dict(provided_contexts, _functionMjs.identity, _functionMjs.identity)
        ],
        [
            "vdom",
            _vnodeMjs.to_json(vdom, memos)
        ]
    ]));
}
function reconcile_to_json(kind, patch, memos) {
    return _jsonMjs.object((0, _gleamMjs.toList)([
        [
            "kind",
            _jsonMjs.int(kind)
        ],
        [
            "patch",
            _patchMjs.to_json(patch, memos)
        ]
    ]));
}
function emit_to_json(kind, name, data) {
    return _jsonMjs.object((0, _gleamMjs.toList)([
        [
            "kind",
            _jsonMjs.int(kind)
        ],
        [
            "name",
            _jsonMjs.string(name)
        ],
        [
            "data",
            data
        ]
    ]));
}
function provide_to_json(kind, key, value) {
    return _jsonMjs.object((0, _gleamMjs.toList)([
        [
            "kind",
            _jsonMjs.int(kind)
        ],
        [
            "key",
            _jsonMjs.string(key)
        ],
        [
            "value",
            value
        ]
    ]));
}
function client_message_to_json(message) {
    if (message instanceof Mount) {
        let kind = message.kind;
        let open_shadow_root = message.open_shadow_root;
        let will_adopt_styles = message.will_adopt_styles;
        let observed_attributes = message.observed_attributes;
        let observed_properties = message.observed_properties;
        let requested_contexts = message.requested_contexts;
        let provided_contexts = message.provided_contexts;
        let vdom = message.vdom;
        let memos = message.memos;
        return mount_to_json(kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos);
    } else if (message instanceof Reconcile) {
        let kind = message.kind;
        let patch = message.patch;
        let memos = message.memos;
        return reconcile_to_json(kind, patch, memos);
    } else if (message instanceof Emit) {
        let kind = message.kind;
        let name = message.name;
        let data = message.data;
        return emit_to_json(kind, name, data);
    } else {
        let kind = message.kind;
        let key = message.key;
        let value = message.value;
        return provide_to_json(kind, key, value);
    }
}
function mount(open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos) {
    return new Mount(mount_kind, open_shadow_root, will_adopt_styles, observed_attributes, observed_properties, requested_contexts, provided_contexts, vdom, memos);
}
function reconcile(patch, memos) {
    return new Reconcile(reconcile_kind, patch, memos);
}
function emit(name, data) {
    return new Emit(emit_kind, name, data);
}
function provide(key, value) {
    return new Provide(provide_kind, key, value);
}
function attribute_changed(name, value) {
    return new AttributeChanged(attribute_changed_kind, name, value);
}
function attribute_changed_decoder() {
    return _decodeMjs.field("name", _decodeMjs.string, (name)=>{
        return _decodeMjs.field("value", _decodeMjs.string, (value)=>{
            return _decodeMjs.success(attribute_changed(name, value));
        });
    });
}
function event_fired(path, name, event) {
    return new EventFired(event_fired_kind, path, name, event);
}
function event_fired_decoder() {
    return _decodeMjs.field("path", _decodeMjs.string, (path)=>{
        return _decodeMjs.field("name", _decodeMjs.string, (name)=>{
            return _decodeMjs.field("event", _decodeMjs.dynamic, (event)=>{
                return _decodeMjs.success(event_fired(path, name, event));
            });
        });
    });
}
function property_changed(name, value) {
    return new PropertyChanged(property_changed_kind, name, value);
}
function property_changed_decoder() {
    return _decodeMjs.field("name", _decodeMjs.string, (name)=>{
        return _decodeMjs.field("value", _decodeMjs.dynamic, (value)=>{
            return _decodeMjs.success(property_changed(name, value));
        });
    });
}
function batch(messages) {
    return new Batch(batch_kind, messages);
}
function context_provided(key, value) {
    return new ContextProvided(context_provided_kind, key, value);
}
function context_provided_decoder() {
    return _decodeMjs.field("key", _decodeMjs.string, (key)=>{
        return _decodeMjs.field("value", _decodeMjs.dynamic, (value)=>{
            return _decodeMjs.success(context_provided(key, value));
        });
    });
}
function batch_decoder() {
    return _decodeMjs.field("messages", _decodeMjs.list(server_message_decoder()), (messages)=>{
        return _decodeMjs.success(batch(messages));
    });
}
function server_message_decoder() {
    return _decodeMjs.field("kind", _decodeMjs.int, (kind)=>{
        if (kind === 0) return attribute_changed_decoder();
        else if (kind === 2) return property_changed_decoder();
        else if (kind === 1) return event_fired_decoder();
        else if (kind === 3) return batch_decoder();
        else return _decodeMjs.failure(batch((0, _gleamMjs.toList)([])), "");
    });
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../../gleam_stdlib/gleam/dict.mjs":"b8yrU","../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../gleam_stdlib/gleam/function.mjs":"2jh6y","../../gleam.mjs":"jNPQG","../../lustre/vdom/patch.mjs":"31vMv","../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"31vMv":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Patch", ()=>Patch);
parcelHelpers.export(exports, "Patch$Patch", ()=>Patch$Patch);
parcelHelpers.export(exports, "Patch$isPatch", ()=>Patch$isPatch);
parcelHelpers.export(exports, "Patch$Patch$index", ()=>Patch$Patch$index);
parcelHelpers.export(exports, "Patch$Patch$0", ()=>Patch$Patch$0);
parcelHelpers.export(exports, "Patch$Patch$removed", ()=>Patch$Patch$removed);
parcelHelpers.export(exports, "Patch$Patch$1", ()=>Patch$Patch$1);
parcelHelpers.export(exports, "Patch$Patch$changes", ()=>Patch$Patch$changes);
parcelHelpers.export(exports, "Patch$Patch$2", ()=>Patch$Patch$2);
parcelHelpers.export(exports, "Patch$Patch$children", ()=>Patch$Patch$children);
parcelHelpers.export(exports, "Patch$Patch$3", ()=>Patch$Patch$3);
parcelHelpers.export(exports, "ReplaceText", ()=>ReplaceText);
parcelHelpers.export(exports, "Change$ReplaceText", ()=>Change$ReplaceText);
parcelHelpers.export(exports, "Change$isReplaceText", ()=>Change$isReplaceText);
parcelHelpers.export(exports, "Change$ReplaceText$kind", ()=>Change$ReplaceText$kind);
parcelHelpers.export(exports, "Change$ReplaceText$0", ()=>Change$ReplaceText$0);
parcelHelpers.export(exports, "Change$ReplaceText$content", ()=>Change$ReplaceText$content);
parcelHelpers.export(exports, "Change$ReplaceText$1", ()=>Change$ReplaceText$1);
parcelHelpers.export(exports, "ReplaceInnerHtml", ()=>ReplaceInnerHtml);
parcelHelpers.export(exports, "Change$ReplaceInnerHtml", ()=>Change$ReplaceInnerHtml);
parcelHelpers.export(exports, "Change$isReplaceInnerHtml", ()=>Change$isReplaceInnerHtml);
parcelHelpers.export(exports, "Change$ReplaceInnerHtml$kind", ()=>Change$ReplaceInnerHtml$kind);
parcelHelpers.export(exports, "Change$ReplaceInnerHtml$0", ()=>Change$ReplaceInnerHtml$0);
parcelHelpers.export(exports, "Change$ReplaceInnerHtml$inner_html", ()=>Change$ReplaceInnerHtml$inner_html);
parcelHelpers.export(exports, "Change$ReplaceInnerHtml$1", ()=>Change$ReplaceInnerHtml$1);
parcelHelpers.export(exports, "Update", ()=>Update);
parcelHelpers.export(exports, "Change$Update", ()=>Change$Update);
parcelHelpers.export(exports, "Change$isUpdate", ()=>Change$isUpdate);
parcelHelpers.export(exports, "Change$Update$kind", ()=>Change$Update$kind);
parcelHelpers.export(exports, "Change$Update$0", ()=>Change$Update$0);
parcelHelpers.export(exports, "Change$Update$added", ()=>Change$Update$added);
parcelHelpers.export(exports, "Change$Update$1", ()=>Change$Update$1);
parcelHelpers.export(exports, "Change$Update$removed", ()=>Change$Update$removed);
parcelHelpers.export(exports, "Change$Update$2", ()=>Change$Update$2);
/**
 * Move a keyed child so that it is before the child at the given index.
 */ parcelHelpers.export(exports, "Move", ()=>Move);
parcelHelpers.export(exports, "Change$Move", ()=>Change$Move);
parcelHelpers.export(exports, "Change$isMove", ()=>Change$isMove);
parcelHelpers.export(exports, "Change$Move$kind", ()=>Change$Move$kind);
parcelHelpers.export(exports, "Change$Move$0", ()=>Change$Move$0);
parcelHelpers.export(exports, "Change$Move$key", ()=>Change$Move$key);
parcelHelpers.export(exports, "Change$Move$1", ()=>Change$Move$1);
parcelHelpers.export(exports, "Change$Move$before", ()=>Change$Move$before);
parcelHelpers.export(exports, "Change$Move$2", ()=>Change$Move$2);
/**
 * Replace a node at the given index with a new vnode.
 */ parcelHelpers.export(exports, "Replace", ()=>Replace);
parcelHelpers.export(exports, "Change$Replace", ()=>Change$Replace);
parcelHelpers.export(exports, "Change$isReplace", ()=>Change$isReplace);
parcelHelpers.export(exports, "Change$Replace$kind", ()=>Change$Replace$kind);
parcelHelpers.export(exports, "Change$Replace$0", ()=>Change$Replace$0);
parcelHelpers.export(exports, "Change$Replace$index", ()=>Change$Replace$index);
parcelHelpers.export(exports, "Change$Replace$1", ()=>Change$Replace$1);
parcelHelpers.export(exports, "Change$Replace$with", ()=>Change$Replace$with);
parcelHelpers.export(exports, "Change$Replace$2", ()=>Change$Replace$2);
/**
 * Remove a child at the given index.
 */ parcelHelpers.export(exports, "Remove", ()=>Remove);
parcelHelpers.export(exports, "Change$Remove", ()=>Change$Remove);
parcelHelpers.export(exports, "Change$isRemove", ()=>Change$isRemove);
parcelHelpers.export(exports, "Change$Remove$kind", ()=>Change$Remove$kind);
parcelHelpers.export(exports, "Change$Remove$0", ()=>Change$Remove$0);
parcelHelpers.export(exports, "Change$Remove$index", ()=>Change$Remove$index);
parcelHelpers.export(exports, "Change$Remove$1", ()=>Change$Remove$1);
/**
 * Insert one or multiple children before the child with the given index.
 */ parcelHelpers.export(exports, "Insert", ()=>Insert);
parcelHelpers.export(exports, "Change$Insert", ()=>Change$Insert);
parcelHelpers.export(exports, "Change$isInsert", ()=>Change$isInsert);
parcelHelpers.export(exports, "Change$Insert$kind", ()=>Change$Insert$kind);
parcelHelpers.export(exports, "Change$Insert$0", ()=>Change$Insert$0);
parcelHelpers.export(exports, "Change$Insert$children", ()=>Change$Insert$children);
parcelHelpers.export(exports, "Change$Insert$1", ()=>Change$Insert$1);
parcelHelpers.export(exports, "Change$Insert$before", ()=>Change$Insert$before);
parcelHelpers.export(exports, "Change$Insert$2", ()=>Change$Insert$2);
parcelHelpers.export(exports, "Change$kind", ()=>Change$kind);
parcelHelpers.export(exports, "replace_text_kind", ()=>replace_text_kind);
parcelHelpers.export(exports, "replace_inner_html_kind", ()=>replace_inner_html_kind);
parcelHelpers.export(exports, "update_kind", ()=>update_kind);
parcelHelpers.export(exports, "move_kind", ()=>move_kind);
parcelHelpers.export(exports, "remove_kind", ()=>remove_kind);
parcelHelpers.export(exports, "replace_kind", ()=>replace_kind);
parcelHelpers.export(exports, "insert_kind", ()=>insert_kind);
parcelHelpers.export(exports, "new$", ()=>new$);
parcelHelpers.export(exports, "is_empty", ()=>is_empty);
parcelHelpers.export(exports, "add_child", ()=>add_child);
parcelHelpers.export(exports, "to_json", ()=>to_json);
parcelHelpers.export(exports, "replace_text", ()=>replace_text);
parcelHelpers.export(exports, "replace_inner_html", ()=>replace_inner_html);
parcelHelpers.export(exports, "update", ()=>update);
parcelHelpers.export(exports, "move", ()=>move);
parcelHelpers.export(exports, "remove", ()=>remove);
parcelHelpers.export(exports, "replace", ()=>replace);
parcelHelpers.export(exports, "insert", ()=>insert);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _jsonObjectBuilderMjs = require("../../lustre/internals/json_object_builder.mjs");
var _vattrMjs = require("../../lustre/vdom/vattr.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
class Patch extends (0, _gleamMjs.CustomType) {
    constructor(index, removed, changes, children){
        super();
        this.index = index;
        this.removed = removed;
        this.changes = changes;
        this.children = children;
    }
}
const Patch$Patch = (index, removed, changes, children)=>new Patch(index, removed, changes, children);
const Patch$isPatch = (value)=>value instanceof Patch;
const Patch$Patch$index = (value)=>value.index;
const Patch$Patch$0 = (value)=>value.index;
const Patch$Patch$removed = (value)=>value.removed;
const Patch$Patch$1 = (value)=>value.removed;
const Patch$Patch$changes = (value)=>value.changes;
const Patch$Patch$2 = (value)=>value.changes;
const Patch$Patch$children = (value)=>value.children;
const Patch$Patch$3 = (value)=>value.children;
class ReplaceText extends (0, _gleamMjs.CustomType) {
    constructor(kind, content){
        super();
        this.kind = kind;
        this.content = content;
    }
}
const Change$ReplaceText = (kind, content)=>new ReplaceText(kind, content);
const Change$isReplaceText = (value)=>value instanceof ReplaceText;
const Change$ReplaceText$kind = (value)=>value.kind;
const Change$ReplaceText$0 = (value)=>value.kind;
const Change$ReplaceText$content = (value)=>value.content;
const Change$ReplaceText$1 = (value)=>value.content;
class ReplaceInnerHtml extends (0, _gleamMjs.CustomType) {
    constructor(kind, inner_html){
        super();
        this.kind = kind;
        this.inner_html = inner_html;
    }
}
const Change$ReplaceInnerHtml = (kind, inner_html)=>new ReplaceInnerHtml(kind, inner_html);
const Change$isReplaceInnerHtml = (value)=>value instanceof ReplaceInnerHtml;
const Change$ReplaceInnerHtml$kind = (value)=>value.kind;
const Change$ReplaceInnerHtml$0 = (value)=>value.kind;
const Change$ReplaceInnerHtml$inner_html = (value)=>value.inner_html;
const Change$ReplaceInnerHtml$1 = (value)=>value.inner_html;
class Update extends (0, _gleamMjs.CustomType) {
    constructor(kind, added, removed){
        super();
        this.kind = kind;
        this.added = added;
        this.removed = removed;
    }
}
const Change$Update = (kind, added, removed)=>new Update(kind, added, removed);
const Change$isUpdate = (value)=>value instanceof Update;
const Change$Update$kind = (value)=>value.kind;
const Change$Update$0 = (value)=>value.kind;
const Change$Update$added = (value)=>value.added;
const Change$Update$1 = (value)=>value.added;
const Change$Update$removed = (value)=>value.removed;
const Change$Update$2 = (value)=>value.removed;
class Move extends (0, _gleamMjs.CustomType) {
    constructor(kind, key, before){
        super();
        this.kind = kind;
        this.key = key;
        this.before = before;
    }
}
const Change$Move = (kind, key, before)=>new Move(kind, key, before);
const Change$isMove = (value)=>value instanceof Move;
const Change$Move$kind = (value)=>value.kind;
const Change$Move$0 = (value)=>value.kind;
const Change$Move$key = (value)=>value.key;
const Change$Move$1 = (value)=>value.key;
const Change$Move$before = (value)=>value.before;
const Change$Move$2 = (value)=>value.before;
class Replace extends (0, _gleamMjs.CustomType) {
    constructor(kind, index, with$){
        super();
        this.kind = kind;
        this.index = index;
        this.with = with$;
    }
}
const Change$Replace = (kind, index, with$)=>new Replace(kind, index, with$);
const Change$isReplace = (value)=>value instanceof Replace;
const Change$Replace$kind = (value)=>value.kind;
const Change$Replace$0 = (value)=>value.kind;
const Change$Replace$index = (value)=>value.index;
const Change$Replace$1 = (value)=>value.index;
const Change$Replace$with = (value)=>value.with;
const Change$Replace$2 = (value)=>value.with;
class Remove extends (0, _gleamMjs.CustomType) {
    constructor(kind, index){
        super();
        this.kind = kind;
        this.index = index;
    }
}
const Change$Remove = (kind, index)=>new Remove(kind, index);
const Change$isRemove = (value)=>value instanceof Remove;
const Change$Remove$kind = (value)=>value.kind;
const Change$Remove$0 = (value)=>value.kind;
const Change$Remove$index = (value)=>value.index;
const Change$Remove$1 = (value)=>value.index;
class Insert extends (0, _gleamMjs.CustomType) {
    constructor(kind, children, before){
        super();
        this.kind = kind;
        this.children = children;
        this.before = before;
    }
}
const Change$Insert = (kind, children, before)=>new Insert(kind, children, before);
const Change$isInsert = (value)=>value instanceof Insert;
const Change$Insert$kind = (value)=>value.kind;
const Change$Insert$0 = (value)=>value.kind;
const Change$Insert$children = (value)=>value.children;
const Change$Insert$1 = (value)=>value.children;
const Change$Insert$before = (value)=>value.before;
const Change$Insert$2 = (value)=>value.before;
const Change$kind = (value)=>value.kind;
const replace_text_kind = 0;
const replace_inner_html_kind = 1;
const update_kind = 2;
const move_kind = 3;
const remove_kind = 4;
const replace_kind = 5;
const insert_kind = 6;
function new$(index, removed, changes, children) {
    return new Patch(index, removed, changes, children);
}
function is_empty(patch) {
    let $ = patch.changes;
    if ($ instanceof (0, _gleamMjs.Empty)) {
        let $1 = patch.children;
        if ($1 instanceof (0, _gleamMjs.Empty)) {
            let $2 = patch.removed;
            if ($2 === 0) return true;
            else return false;
        } else return false;
    } else return false;
}
function add_child(parent, child) {
    let $ = is_empty(child);
    if ($) return parent;
    else return new Patch(parent.index, parent.removed, parent.changes, (0, _gleamMjs.prepend)(child, parent.children));
}
function replace_text_to_json(kind, content) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "content", content);
    return _jsonObjectBuilderMjs.build(_pipe$1);
}
function replace_inner_html_to_json(kind, inner_html) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "inner_html", inner_html);
    return _jsonObjectBuilderMjs.build(_pipe$1);
}
function update_to_json(kind, added, removed) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.list(_pipe, "added", added, _vattrMjs.to_json);
    let _pipe$2 = _jsonObjectBuilderMjs.list(_pipe$1, "removed", removed, _vattrMjs.to_json);
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function move_to_json(kind, key, before) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.string(_pipe, "key", key);
    let _pipe$2 = _jsonObjectBuilderMjs.int(_pipe$1, "before", before);
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function remove_to_json(kind, index) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.int(_pipe, "index", index);
    return _jsonObjectBuilderMjs.build(_pipe$1);
}
function replace_to_json(kind, index, with$, memos) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.int(_pipe, "index", index);
    let _pipe$2 = _jsonObjectBuilderMjs.json(_pipe$1, "with", _vnodeMjs.to_json(with$, memos));
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function insert_to_json(kind, children, before, memos) {
    let _pipe = _jsonObjectBuilderMjs.tagged(kind);
    let _pipe$1 = _jsonObjectBuilderMjs.int(_pipe, "before", before);
    let _pipe$2 = _jsonObjectBuilderMjs.list(_pipe$1, "children", children, (_capture)=>{
        return _vnodeMjs.to_json(_capture, memos);
    });
    return _jsonObjectBuilderMjs.build(_pipe$2);
}
function change_to_json(change, memos) {
    if (change instanceof ReplaceText) {
        let kind = change.kind;
        let content = change.content;
        return replace_text_to_json(kind, content);
    } else if (change instanceof ReplaceInnerHtml) {
        let kind = change.kind;
        let inner_html = change.inner_html;
        return replace_inner_html_to_json(kind, inner_html);
    } else if (change instanceof Update) {
        let kind = change.kind;
        let added = change.added;
        let removed = change.removed;
        return update_to_json(kind, added, removed);
    } else if (change instanceof Move) {
        let kind = change.kind;
        let key = change.key;
        let before = change.before;
        return move_to_json(kind, key, before);
    } else if (change instanceof Replace) {
        let kind = change.kind;
        let index = change.index;
        let with$ = change.with;
        return replace_to_json(kind, index, with$, memos);
    } else if (change instanceof Remove) {
        let kind = change.kind;
        let index = change.index;
        return remove_to_json(kind, index);
    } else {
        let kind = change.kind;
        let children = change.children;
        let before = change.before;
        return insert_to_json(kind, children, before, memos);
    }
}
function to_json(patch, memos) {
    let _pipe = _jsonObjectBuilderMjs.new$();
    let _pipe$1 = _jsonObjectBuilderMjs.int(_pipe, "index", patch.index);
    let _pipe$2 = _jsonObjectBuilderMjs.int(_pipe$1, "removed", patch.removed);
    let _pipe$3 = _jsonObjectBuilderMjs.list(_pipe$2, "changes", patch.changes, (change)=>{
        return change_to_json(change, memos);
    });
    let _pipe$4 = _jsonObjectBuilderMjs.list(_pipe$3, "children", patch.children, (child)=>{
        return to_json(child, memos);
    });
    return _jsonObjectBuilderMjs.build(_pipe$4);
}
function replace_text(content) {
    return new ReplaceText(replace_text_kind, content);
}
function replace_inner_html(inner_html) {
    return new ReplaceInnerHtml(replace_inner_html_kind, inner_html);
}
function update(added, removed) {
    return new Update(update_kind, added, removed);
}
function move(key, before) {
    return new Move(move_kind, key, before);
}
function remove(index) {
    return new Remove(remove_kind, index);
}
function replace(index, with$) {
    return new Replace(replace_kind, index, with$);
}
function insert(children, before) {
    return new Insert(insert_kind, children, before);
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam.mjs":"jNPQG","../../lustre/internals/json_object_builder.mjs":"31ZqD","../../lustre/vdom/vattr.mjs":"jrrcC","../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"aEh50":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "compose_mapper", ()=>compose_mapper);
parcelHelpers.export(exports, "new_events", ()=>new_events);
/**
 *
 */ parcelHelpers.export(exports, "new$", ()=>new$);
parcelHelpers.export(exports, "tick", ()=>tick);
parcelHelpers.export(exports, "events", ()=>events);
parcelHelpers.export(exports, "update_events", ()=>update_events);
/**
 * Get a dictionary of all materialised Memo views.
 */ parcelHelpers.export(exports, "memos", ()=>memos);
/**
 *
 */ parcelHelpers.export(exports, "get_old_memo", ()=>get_old_memo);
/**
 * Reuses the cached element when dependencies are unchanged.
 */ parcelHelpers.export(exports, "keep_memo", ()=>keep_memo);
/**
 * Caches a newly computed element when dependencies changed.
 */ parcelHelpers.export(exports, "add_memo", ()=>add_memo);
/**
 * Gets the isolated event subtree for a Map node.
 */ parcelHelpers.export(exports, "get_subtree", ()=>get_subtree);
/**
 * Updates the Map node's isolated event subtree after diffing its child.
 */ parcelHelpers.export(exports, "update_subtree", ()=>update_subtree);
parcelHelpers.export(exports, "add_event", ()=>add_event);
parcelHelpers.export(exports, "remove_event", ()=>remove_event);
/**
 *
 */ parcelHelpers.export(exports, "add_children", ()=>add_children);
parcelHelpers.export(exports, "add_child", ()=>add_child);
parcelHelpers.export(exports, "from_node", ()=>from_node);
parcelHelpers.export(exports, "remove_child", ()=>remove_child);
parcelHelpers.export(exports, "replace_child", ()=>replace_child);
parcelHelpers.export(exports, "dispatch", ()=>dispatch);
parcelHelpers.export(exports, "has_dispatched_events", ()=>has_dispatched_events);
parcelHelpers.export(exports, "decode", ()=>decode);
/**
 *
 */ parcelHelpers.export(exports, "handle", ()=>handle);
var _dynamicMjs = require("../../../gleam_stdlib/gleam/dynamic.mjs");
var _decodeMjs = require("../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _functionMjs = require("../../../gleam_stdlib/gleam/function.mjs");
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
var _mutableMapMjs = require("../../lustre/internals/mutable_map.mjs");
var _pathMjs = require("../../lustre/vdom/path.mjs");
var _vattrMjs = require("../../lustre/vdom/vattr.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
class Cache extends (0, _gleamMjs.CustomType) {
    constructor(events, vdoms, old_vdoms, dispatched_paths, next_dispatched_paths){
        super();
        this.events = events;
        this.vdoms = vdoms;
        this.old_vdoms = old_vdoms;
        this.dispatched_paths = dispatched_paths;
        this.next_dispatched_paths = next_dispatched_paths;
    }
}
class Events extends (0, _gleamMjs.CustomType) {
    constructor(handlers, children){
        super();
        this.handlers = handlers;
        this.children = children;
    }
}
class Child extends (0, _gleamMjs.CustomType) {
    constructor(mapper, events){
        super();
        this.mapper = mapper;
        this.events = events;
    }
}
class AddedChildren extends (0, _gleamMjs.CustomType) {
    constructor(handlers, children, vdoms){
        super();
        this.handlers = handlers;
        this.children = children;
        this.vdoms = vdoms;
    }
}
class DecodedEvent extends (0, _gleamMjs.CustomType) {
    constructor(path, handler){
        super();
        this.path = path;
        this.handler = handler;
    }
}
class DispatchedEvent extends (0, _gleamMjs.CustomType) {
    constructor(path){
        super();
        this.path = path;
    }
}
function compose_mapper(mapper, child_mapper) {
    return (msg)=>{
        return mapper(child_mapper(msg));
    };
}
function new_events() {
    return new Events(_mutableMapMjs.new$(), _mutableMapMjs.new$());
}
function new$() {
    return new Cache(new_events(), _mutableMapMjs.new$(), _mutableMapMjs.new$(), _constantsMjs.empty_list, _constantsMjs.empty_list);
}
function tick(cache) {
    return new Cache(cache.events, _mutableMapMjs.new$(), cache.vdoms, cache.next_dispatched_paths, _constantsMjs.empty_list);
}
function events(cache) {
    return cache.events;
}
function update_events(cache, events) {
    return new Cache(events, cache.vdoms, cache.old_vdoms, cache.dispatched_paths, cache.next_dispatched_paths);
}
function memos(cache) {
    return cache.vdoms;
}
function get_old_memo(cache, old, new$) {
    return _mutableMapMjs.get_or_compute(cache.old_vdoms, old, new$);
}
function keep_memo(cache, old, new$) {
    let node = _mutableMapMjs.get_or_compute(cache.old_vdoms, old, new$);
    let vdoms = _mutableMapMjs.insert(cache.vdoms, new$, node);
    return new Cache(cache.events, vdoms, cache.old_vdoms, cache.dispatched_paths, cache.next_dispatched_paths);
}
function add_memo(cache, new$, node) {
    let vdoms = _mutableMapMjs.insert(cache.vdoms, new$, node);
    return new Cache(cache.events, vdoms, cache.old_vdoms, cache.dispatched_paths, cache.next_dispatched_paths);
}
function get_subtree(events, path, old_mapper) {
    let child = _mutableMapMjs.get_or_compute(events.children, path, ()=>{
        return new Child(old_mapper, new_events());
    });
    return child.events;
}
function update_subtree(parent, path, mapper, events) {
    let new_child = new Child(mapper, events);
    let children = _mutableMapMjs.insert(parent.children, path, new_child);
    return new Events(parent.handlers, children);
}
function do_add_event(handlers, path, name, handler) {
    return _mutableMapMjs.insert(handlers, _pathMjs.event(path, name), handler);
}
function add_event(events, path, name, handler) {
    let handlers = do_add_event(events.handlers, path, name, handler);
    return new Events(handlers, events.children);
}
function do_remove_event(handlers, path, name) {
    return _mutableMapMjs.delete$(handlers, _pathMjs.event(path, name));
}
function remove_event(events, path, name) {
    let handlers = do_remove_event(events.handlers, path, name);
    return new Events(handlers, events.children);
}
function add_attributes(handlers, path, attributes) {
    return _listMjs.fold(attributes, handlers, (events, attribute)=>{
        if (attribute instanceof (0, _vattrMjs.Event)) {
            let name = attribute.name;
            let handler = attribute.handler;
            return do_add_event(events, path, name, handler);
        } else return events;
    });
}
function do_add_children(loop$handlers, loop$children, loop$vdoms, loop$parent, loop$child_index, loop$nodes) {
    while(true){
        let handlers = loop$handlers;
        let children = loop$children;
        let vdoms = loop$vdoms;
        let parent = loop$parent;
        let child_index = loop$child_index;
        let nodes = loop$nodes;
        let next = child_index + 1;
        if (nodes instanceof (0, _gleamMjs.Empty)) return new AddedChildren(handlers, children, vdoms);
        else {
            let $ = nodes.head;
            if ($ instanceof (0, _vnodeMjs.Fragment)) {
                let rest = nodes.tail;
                let key = $.key;
                let nodes$1 = $.children;
                let path = _pathMjs.add(parent, child_index, key);
                let $1 = do_add_children(handlers, children, vdoms, path, 0, nodes$1);
                let handlers$1;
                let children$1;
                let vdoms$1;
                handlers$1 = $1.handlers;
                children$1 = $1.children;
                vdoms$1 = $1.vdoms;
                loop$handlers = handlers$1;
                loop$children = children$1;
                loop$vdoms = vdoms$1;
                loop$parent = parent;
                loop$child_index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Element)) {
                let rest = nodes.tail;
                let key = $.key;
                let attributes = $.attributes;
                let nodes$1 = $.children;
                let path = _pathMjs.add(parent, child_index, key);
                let handlers$1 = add_attributes(handlers, path, attributes);
                let $1 = do_add_children(handlers$1, children, vdoms, path, 0, nodes$1);
                let handlers$2;
                let children$1;
                let vdoms$1;
                handlers$2 = $1.handlers;
                children$1 = $1.children;
                vdoms$1 = $1.vdoms;
                loop$handlers = handlers$2;
                loop$children = children$1;
                loop$vdoms = vdoms$1;
                loop$parent = parent;
                loop$child_index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Text)) {
                let rest = nodes.tail;
                loop$handlers = handlers;
                loop$children = children;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$child_index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.UnsafeInnerHtml)) {
                let rest = nodes.tail;
                let key = $.key;
                let attributes = $.attributes;
                let path = _pathMjs.add(parent, child_index, key);
                let handlers$1 = add_attributes(handlers, path, attributes);
                loop$handlers = handlers$1;
                loop$children = children;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$child_index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Map)) {
                let rest = nodes.tail;
                let key = $.key;
                let mapper = $.mapper;
                let child = $.child;
                let path = _pathMjs.add(parent, child_index, key);
                let added = do_add_children(_mutableMapMjs.new$(), _mutableMapMjs.new$(), vdoms, _pathMjs.subtree(path), 0, (0, _gleamMjs.prepend)(child, _constantsMjs.empty_list));
                let vdoms$1 = added.vdoms;
                let child_events = new Events(added.handlers, added.children);
                let child$1 = new Child(mapper, child_events);
                let children$1 = _mutableMapMjs.insert(children, _pathMjs.child(path), child$1);
                loop$handlers = handlers;
                loop$children = children$1;
                loop$vdoms = vdoms$1;
                loop$parent = parent;
                loop$child_index = next;
                loop$nodes = rest;
            } else {
                let rest = nodes.tail;
                let view = $.view;
                let child_node = view();
                let vdoms$1 = _mutableMapMjs.insert(vdoms, view, child_node);
                let next$1 = child_index;
                let rest$1 = (0, _gleamMjs.prepend)(child_node, rest);
                loop$handlers = handlers;
                loop$children = children;
                loop$vdoms = vdoms$1;
                loop$parent = parent;
                loop$child_index = next$1;
                loop$nodes = rest$1;
            }
        }
    }
}
function add_children(cache, events, path, child_index, nodes) {
    let vdoms = cache.vdoms;
    let handlers;
    let children;
    handlers = events.handlers;
    children = events.children;
    let $ = do_add_children(handlers, children, vdoms, path, child_index, nodes);
    let handlers$1;
    let children$1;
    let vdoms$1;
    handlers$1 = $.handlers;
    children$1 = $.children;
    vdoms$1 = $.vdoms;
    return [
        new Cache(cache.events, vdoms$1, cache.old_vdoms, cache.dispatched_paths, cache.next_dispatched_paths),
        new Events(handlers$1, children$1)
    ];
}
function add_child(cache, events, parent, index, child) {
    let children = (0, _gleamMjs.prepend)(child, _constantsMjs.empty_list);
    return add_children(cache, events, parent, index, children);
}
function from_node(root) {
    let cache = new$();
    let $ = add_child(cache, cache.events, _pathMjs.root, 0, root);
    let cache$1;
    let events$1;
    cache$1 = $[0];
    events$1 = $[1];
    return new Cache(events$1, cache$1.vdoms, cache$1.old_vdoms, cache$1.dispatched_paths, cache$1.next_dispatched_paths);
}
function remove_attributes(handlers, path, attributes) {
    return _listMjs.fold(attributes, handlers, (events, attribute)=>{
        if (attribute instanceof (0, _vattrMjs.Event)) {
            let name = attribute.name;
            return do_remove_event(events, path, name);
        } else return events;
    });
}
function do_remove_children(loop$handlers, loop$children, loop$vdoms, loop$parent, loop$index, loop$nodes) {
    while(true){
        let handlers = loop$handlers;
        let children = loop$children;
        let vdoms = loop$vdoms;
        let parent = loop$parent;
        let index = loop$index;
        let nodes = loop$nodes;
        let next = index + 1;
        if (nodes instanceof (0, _gleamMjs.Empty)) return new Events(handlers, children);
        else {
            let $ = nodes.head;
            if ($ instanceof (0, _vnodeMjs.Fragment)) {
                let rest = nodes.tail;
                let key = $.key;
                let nodes$1 = $.children;
                let path = _pathMjs.add(parent, index, key);
                let $1 = do_remove_children(handlers, children, vdoms, path, 0, nodes$1);
                let handlers$1;
                let children$1;
                handlers$1 = $1.handlers;
                children$1 = $1.children;
                loop$handlers = handlers$1;
                loop$children = children$1;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Element)) {
                let rest = nodes.tail;
                let key = $.key;
                let attributes = $.attributes;
                let nodes$1 = $.children;
                let path = _pathMjs.add(parent, index, key);
                let handlers$1 = remove_attributes(handlers, path, attributes);
                let $1 = do_remove_children(handlers$1, children, vdoms, path, 0, nodes$1);
                let handlers$2;
                let children$1;
                handlers$2 = $1.handlers;
                children$1 = $1.children;
                loop$handlers = handlers$2;
                loop$children = children$1;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Text)) {
                let rest = nodes.tail;
                loop$handlers = handlers;
                loop$children = children;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.UnsafeInnerHtml)) {
                let rest = nodes.tail;
                let key = $.key;
                let attributes = $.attributes;
                let path = _pathMjs.add(parent, index, key);
                let handlers$1 = remove_attributes(handlers, path, attributes);
                loop$handlers = handlers$1;
                loop$children = children;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$index = next;
                loop$nodes = rest;
            } else if ($ instanceof (0, _vnodeMjs.Map)) {
                let rest = nodes.tail;
                let key = $.key;
                let path = _pathMjs.add(parent, index, key);
                let children$1 = _mutableMapMjs.delete$(children, _pathMjs.child(path));
                loop$handlers = handlers;
                loop$children = children$1;
                loop$vdoms = vdoms;
                loop$parent = parent;
                loop$index = next;
                loop$nodes = rest;
            } else {
                let rest = nodes.tail;
                let view = $.view;
                let $1 = _mutableMapMjs.has_key(vdoms, view);
                if ($1) {
                    let child = _mutableMapMjs.unsafe_get(vdoms, view);
                    let nodes$1 = (0, _gleamMjs.prepend)(child, rest);
                    loop$handlers = handlers;
                    loop$children = children;
                    loop$vdoms = vdoms;
                    loop$parent = parent;
                    loop$index = index;
                    loop$nodes = nodes$1;
                } else {
                    loop$handlers = handlers;
                    loop$children = children;
                    loop$vdoms = vdoms;
                    loop$parent = parent;
                    loop$index = next;
                    loop$nodes = rest;
                }
            }
        }
    }
}
function remove_child(cache, events, parent, child_index, child) {
    return do_remove_children(events.handlers, events.children, cache.old_vdoms, parent, child_index, (0, _gleamMjs.prepend)(child, _constantsMjs.empty_list));
}
function replace_child(cache, events, parent, child_index, prev, next) {
    let events$1 = remove_child(cache, events, parent, child_index, prev);
    return add_child(cache, events$1, parent, child_index, next);
}
function dispatch(cache, event) {
    let next_dispatched_paths = (0, _gleamMjs.prepend)(event.path, cache.next_dispatched_paths);
    let cache$1 = new Cache(cache.events, cache.vdoms, cache.old_vdoms, cache.dispatched_paths, next_dispatched_paths);
    if (event instanceof DecodedEvent) {
        let handler = event.handler;
        return [
            cache$1,
            new (0, _gleamMjs.Ok)(handler)
        ];
    } else return [
        cache$1,
        _constantsMjs.error_nil
    ];
}
function has_dispatched_events(cache, path) {
    return _pathMjs.matches(path, cache.dispatched_paths);
}
function get_handler(loop$events, loop$path, loop$mapper) {
    while(true){
        let events = loop$events;
        let path = loop$path;
        let mapper = loop$mapper;
        if (path instanceof (0, _gleamMjs.Empty)) return _constantsMjs.error_nil;
        else {
            let $ = path.tail;
            if ($ instanceof (0, _gleamMjs.Empty)) {
                let key = path.head;
                let $1 = _mutableMapMjs.has_key(events.handlers, key);
                if ($1) {
                    let handler = _mutableMapMjs.unsafe_get(events.handlers, key);
                    return new (0, _gleamMjs.Ok)(_decodeMjs.map(handler, (handler)=>{
                        return new (0, _vattrMjs.Handler)(handler.prevent_default, handler.stop_propagation, (0, _functionMjs.identity)(mapper)(handler.message));
                    }));
                } else return _constantsMjs.error_nil;
            } else {
                let key = path.head;
                let path$1 = $;
                let $1 = _mutableMapMjs.has_key(events.children, key);
                if ($1) {
                    let child = _mutableMapMjs.unsafe_get(events.children, key);
                    let mapper$1 = compose_mapper(mapper, child.mapper);
                    loop$events = child.events;
                    loop$path = path$1;
                    loop$mapper = mapper$1;
                } else return _constantsMjs.error_nil;
            }
        }
    }
}
function decode(cache, path, name, event) {
    let parts = _pathMjs.split_subtree_path(path + _pathMjs.separator_event + name);
    let $ = get_handler(cache.events, parts, _functionMjs.identity);
    if ($ instanceof (0, _gleamMjs.Ok)) {
        let handler = $[0];
        let $1 = _decodeMjs.run(event, handler);
        if ($1 instanceof (0, _gleamMjs.Ok)) {
            let handler$1 = $1[0];
            return new DecodedEvent(path, handler$1);
        } else return new DispatchedEvent(path);
    } else return new DispatchedEvent(path);
}
function handle(cache, path, name, event) {
    let _pipe = decode(cache, path, name, event);
    return ((_capture)=>{
        return dispatch(cache, _capture);
    })(_pipe);
}

},{"../../../gleam_stdlib/gleam/dynamic.mjs":"iAWCk","../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../gleam_stdlib/gleam/function.mjs":"2jh6y","../../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../gleam.mjs":"jNPQG","../../lustre/internals/constants.mjs":"gKFR6","../../lustre/internals/mutable_map.mjs":"6NvMa","../../lustre/vdom/path.mjs":"351yX","../../lustre/vdom/vattr.mjs":"jrrcC","../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"351yX":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "root", ()=>root);
parcelHelpers.export(exports, "separator_element", ()=>separator_element);
parcelHelpers.export(exports, "separator_subtree", ()=>separator_subtree);
parcelHelpers.export(exports, "separator_event", ()=>separator_event);
/**
 *
 */ parcelHelpers.export(exports, "add", ()=>add);
parcelHelpers.export(exports, "subtree", ()=>subtree);
parcelHelpers.export(exports, "split_subtree_path", ()=>split_subtree_path);
/**
 * Convert a path to a child tree to a resolved string.
 */ parcelHelpers.export(exports, "child", ()=>child);
/**
 * Convert a path to a full resolved string, including all memo barriers.
 */ parcelHelpers.export(exports, "to_string", ()=>to_string);
/**
 *
 */ parcelHelpers.export(exports, "matches", ()=>matches);
/**
 * Convert a path to a resolved string with an event name appended to it.
 * This returns a partial path, up to the closest Memo barrier.
 */ parcelHelpers.export(exports, "event", ()=>event);
var _intMjs = require("../../../gleam_stdlib/gleam/int.mjs");
var _stringMjs = require("../../../gleam_stdlib/gleam/string.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
class Root extends (0, _gleamMjs.CustomType) {
}
class Key extends (0, _gleamMjs.CustomType) {
    constructor(key, parent){
        super();
        this.key = key;
        this.parent = parent;
    }
}
class Index extends (0, _gleamMjs.CustomType) {
    constructor(index, parent){
        super();
        this.index = index;
        this.parent = parent;
    }
}
class Subtree extends (0, _gleamMjs.CustomType) {
    constructor(parent){
        super();
        this.parent = parent;
    }
}
const root = /* @__PURE__ */ new Root();
const separator_element = "\t";
const separator_subtree = "\r";
const separator_event = "\n";
function do_matches(loop$path, loop$candidates) {
    while(true){
        let path = loop$path;
        let candidates = loop$candidates;
        if (candidates instanceof (0, _gleamMjs.Empty)) return false;
        else {
            let candidate = candidates.head;
            let rest = candidates.tail;
            let $ = _stringMjs.starts_with(path, candidate);
            if ($) return $;
            else {
                loop$path = path;
                loop$candidates = rest;
            }
        }
    }
}
function add(parent, index, key) {
    if (key === "") return new Index(index, parent);
    else return new Key(key, parent);
}
function subtree(path) {
    return new Subtree(path);
}
function finish_to_string(acc) {
    if (acc instanceof (0, _gleamMjs.Empty)) return "";
    else {
        let segments = acc.tail;
        return _stringMjs.concat(segments);
    }
}
function split_subtree_path(path) {
    return _stringMjs.split(path, separator_subtree);
}
function do_to_string(loop$full, loop$path, loop$acc) {
    while(true){
        let full = loop$full;
        let path = loop$path;
        let acc = loop$acc;
        if (path instanceof Root) return finish_to_string(acc);
        else if (path instanceof Key) {
            let key = path.key;
            let parent = path.parent;
            loop$full = full;
            loop$path = parent;
            loop$acc = (0, _gleamMjs.prepend)(separator_element, (0, _gleamMjs.prepend)(key, acc));
        } else if (path instanceof Index) {
            let index = path.index;
            let parent = path.parent;
            let acc$1 = (0, _gleamMjs.prepend)(separator_element, (0, _gleamMjs.prepend)(_intMjs.to_string(index), acc));
            loop$full = full;
            loop$path = parent;
            loop$acc = acc$1;
        } else if (!full) return finish_to_string(acc);
        else {
            let parent = path.parent;
            if (acc instanceof (0, _gleamMjs.Empty)) {
                loop$full = full;
                loop$path = parent;
                loop$acc = acc;
            } else {
                let acc$1 = acc.tail;
                loop$full = full;
                loop$path = parent;
                loop$acc = (0, _gleamMjs.prepend)(separator_subtree, acc$1);
            }
        }
    }
}
function child(path) {
    return do_to_string(false, path, _constantsMjs.empty_list);
}
function to_string(path) {
    return do_to_string(true, path, _constantsMjs.empty_list);
}
function matches(path, candidates) {
    if (candidates instanceof (0, _gleamMjs.Empty)) return false;
    else return do_matches(to_string(path), candidates);
}
function event(path, event) {
    return do_to_string(false, path, (0, _gleamMjs.prepend)(separator_event, (0, _gleamMjs.prepend)(event, _constantsMjs.empty_list)));
}

},{"../../../gleam_stdlib/gleam/int.mjs":"32hLf","../../../gleam_stdlib/gleam/string.mjs":"aB8qb","../../gleam.mjs":"jNPQG","../../lustre/internals/constants.mjs":"gKFR6","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eGPg4":[function(require,module,exports,__globalThis) {
// IMPORTS ---------------------------------------------------------------------
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "make_component", ()=>make_component);
parcelHelpers.export(exports, "set_form_value", ()=>set_form_value);
parcelHelpers.export(exports, "clear_form_value", ()=>clear_form_value);
parcelHelpers.export(exports, "set_pseudo_state", ()=>set_pseudo_state);
parcelHelpers.export(exports, "remove_pseudo_state", ()=>remove_pseudo_state);
var _gleamMjs = require("../../../gleam.mjs");
var _decodeMjs = require("../../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _optionMjs = require("../../../../gleam_stdlib/gleam/option.mjs");
var _lustreMjs = require("../../../lustre.mjs");
var _runtimeFfiMjs = require("./runtime.ffi.mjs");
var _runtimeMjs = require("../server/runtime.mjs");
var _listFfiMjs = require("../../internals/list.ffi.mjs");
const make_component = ({ init, update, view, config }, name)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return (0, _gleamMjs.Result$Error)((0, _lustreMjs.Error$NotABrowser)());
    if (!name.includes("-")) return (0, _gleamMjs.Result$Error)((0, _lustreMjs.Error$BadComponentName)(name));
    if (globalThis.customElements.get(name)) return (0, _gleamMjs.Result$Error)((0, _lustreMjs.Error$ComponentAlreadyRegistered)(name));
    const attributes = new Map();
    const observedAttributes = [];
    (0, _listFfiMjs.iterate)(config.attributes, ([name, decoder])=>{
        if (attributes.has(name)) return;
        attributes.set(name, decoder);
        observedAttributes.push(name);
    });
    const [model, effects] = init(undefined);
    const component = class Component extends globalThis.HTMLElement {
        static get observedAttributes() {
            return observedAttributes;
        }
        static formAssociated = config.is_form_associated;
        #runtime;
        #adoptedStyleNodes = [];
        #contextSubscriptions = new Map();
        constructor(){
            super();
            // There are talks of potentially having `attachInternals` set `.internals`
            // automatically in the future.
            this.internals = this.attachInternals();
            // Only attach a shadow root if we don't already have one from the declarative
            // shadow DOM. This means components can be SSR'd and then hydrated like
            // normal apps.
            if (!this.internals.shadowRoot) this.attachShadow({
                mode: config.open_shadow_root ? "open" : "closed",
                delegatesFocus: config.delegates_focus
            });
            if (config.adopt_styles) this.#adoptStyleSheets();
            this.#runtime = new (0, _runtimeFfiMjs.Runtime)(this.internals.shadowRoot, [
                model,
                effects
            ], view, update);
        }
        // CUSTOM ELEMENT LIFECYCLE METHODS ----------------------------------------
        // When an element is constructed by `document.createElement` and then added
        // to the DOM, the lifecycle callbacks run in this order:
        //
        //   constructor -> attributeChangedCallback -> connectedCallback
        //
        // If the element is added to the document through `document.importNode` then
        // we get:
        //
        //   constructor -> connectedCallback
        //
        // The connectedCallback is also called when the element is moved to a new
        // position in the same document, so it's important we don't do any *one-time*
        // work here.
        //
        connectedCallback() {
            this.#requestContexts();
            if ((0, _optionMjs.Option$isSome)(config.on_connect)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_connect));
        }
        // If the element is imported into the document through `document.adoptNode`
        // then the lifecycle callbacks are:
        //
        //   disconnectedCallback -> adoptedCallback -> connectedCallback
        //
        adoptedCallback() {
            if (config.adopt_styles) this.#adoptStyleSheets();
            this.#unsubscribeContexts();
            if ((0, _optionMjs.Option$isSome)(config.on_adopt)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_adopt));
        }
        // The disconnected callback is also called when the element is disconnected
        // from the document even if it is reconnected somewhere else. It's important
        // we use this callback just for DOM-related cleanup.
        //
        disconnectedCallback() {
            this.#unsubscribeContexts();
            if ((0, _optionMjs.Option$isSome)(config.on_disconnect)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_disconnect));
        }
        attributeChangedCallback(name, _, value) {
            const decoded = attributes.get(name)(value ?? "");
            if ((0, _gleamMjs.Result$isOk)(decoded)) this.dispatch((0, _gleamMjs.Result$Ok$0)(decoded), true);
        }
        formResetCallback() {
            if ((0, _optionMjs.Option$isSome)(config.on_form_reset)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_form_reset));
        }
        formStateRestoreCallback(state, reason) {
            switch(reason){
                case "restore":
                    if ((0, _optionMjs.Option$isSome)(config.on_form_restore)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_form_restore)(state));
                    break;
                case "autocomplete":
                    if ((0, _optionMjs.Option$isSome)(config.on_form_autofill)) this.dispatch((0, _optionMjs.Option$Some$0)(config.on_form_autofill)(state));
                    break;
            }
        }
        // LUSTRE RUNTIME METHODS --------------------------------------------------
        send(message) {
            if ((0, _runtimeMjs.Message$isEffectDispatchedMessage)(message)) this.dispatch(message.message, false);
            else if ((0, _runtimeMjs.Message$isEffectEmitEvent)(message)) this.emit(message.name, message.data);
            else (0, _runtimeMjs.Message$isSystemRequestedShutdown)(message);
        }
        dispatch(msg, shouldFlush = false) {
            this.#runtime.dispatch(msg, shouldFlush);
        }
        emit(event, data) {
            this.#runtime.emit(event, data);
        }
        provide(key, value) {
            this.#runtime.provide(key, value);
        }
        // INTERNAL METHODS --------------------------------------------------------
        #requestContexts() {
            const requested = new Set();
            (0, _listFfiMjs.iterate)(config.contexts, ([key, decoder])=>{
                // An empty key is not valid so we skip over any of those.
                if (!key) return;
                // Likewise if we've requested a context for this key already then we
                // don't want to dispatch a second event, even if the user provided a
                // different decoder.
                if (requested.has(key)) return;
                this.dispatchEvent(new (0, _runtimeFfiMjs.ContextRequestEvent)(key, (value, unsubscribe)=>{
                    const previousUnsubscribe = this.#contextSubscriptions.get(key);
                    // Call the old unsubscribe callback if it has changed. This probably
                    // means we have a new provider.
                    if (previousUnsubscribe !== unsubscribe) previousUnsubscribe?.();
                    const decoded = (0, _decodeMjs.run)(value, decoder);
                    this.#contextSubscriptions.set(key, unsubscribe);
                    if ((0, _gleamMjs.Result$isOk)(decoded)) this.dispatch((0, _gleamMjs.Result$Ok$0)(decoded), true);
                }, true));
                requested.add(key);
            });
        }
        #unsubscribeContexts() {
            for (const [_, unsubscribe] of this.#contextSubscriptions)unsubscribe?.();
            this.#contextSubscriptions.clear();
        }
        async #adoptStyleSheets() {
            while(this.#adoptedStyleNodes.length){
                this.#adoptedStyleNodes.pop().remove();
                this.shadowRoot.firstChild.remove();
            }
            this.#adoptedStyleNodes = await (0, _runtimeFfiMjs.adoptStylesheets)(this.internals.shadowRoot);
        }
    };
    (0, _listFfiMjs.iterate)(config.properties, ([name, decoder])=>{
        if (Object.hasOwn(component.prototype, name)) return;
        Object.defineProperty(component.prototype, name, {
            get () {
                return this[`_${name}`];
            },
            set (value) {
                this[`_${name}`] = value;
                const decoded = (0, _decodeMjs.run)(value, decoder);
                if ((0, _gleamMjs.Result$isOk)(decoded)) this.dispatch((0, _gleamMjs.Result$Ok$0)(decoded), true);
            }
        });
    });
    globalThis.customElements.define(name, component);
    return (0, _gleamMjs.Result$Ok)(undefined);
};
const set_form_value = (root, value)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return;
    if (root instanceof ShadowRoot) root.host.internals.setFormValue(value);
};
const clear_form_value = (root)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return;
    if (root instanceof ShadowRoot) root.host.internals.setFormValue(undefined);
};
const set_pseudo_state = (root, value)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return;
    if (root instanceof ShadowRoot) root.host.internals.states.add(value);
};
const remove_pseudo_state = (root, value)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return;
    if (root instanceof ShadowRoot) root.host.internals.states.delete(value);
};

},{"../../../gleam.mjs":"jNPQG","../../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../../lustre.mjs":"9FST8","./runtime.ffi.mjs":"eto4y","../server/runtime.mjs":"8rUwG","../../internals/list.ffi.mjs":"hGVW1","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eto4y":[function(require,module,exports,__globalThis) {
// IMPORTS ---------------------------------------------------------------------
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "is_browser", ()=>is_browser);
parcelHelpers.export(exports, "is_registered", ()=>is_registered);
parcelHelpers.export(exports, "throw_server_component_error", ()=>throw_server_component_error);
//
parcelHelpers.export(exports, "Runtime", ()=>Runtime);
parcelHelpers.export(exports, "send", ()=>send);
parcelHelpers.export(exports, "adoptStylesheets", ()=>adoptStylesheets);
parcelHelpers.export(exports, "ContextRequestEvent", ()=>ContextRequestEvent);
var _gleamMjs = require("../../../gleam.mjs");
var _constantsMjs = require("../../internals/constants.mjs");
var _diffMjs = require("../../vdom/diff.mjs");
var _cacheMjs = require("../../vdom/cache.mjs");
var _reconcilerFfiMjs = require("../../vdom/reconciler.ffi.mjs");
var _virtualiseFfiMjs = require("../../vdom/virtualise.ffi.mjs");
var _equalsFfiMjs = require("../../internals/equals.ffi.mjs");
var _listFfiMjs = require("../../internals/list.ffi.mjs");
const is_browser = ()=>!!globalThis.document;
const is_registered = (name)=>is_browser() && customElements.get(name);
const throw_server_component_error = ()=>{
    throw new globalThis.Error([
        "It looks like you're trying to use the server component runtime written ",
        "using `gleam_otp`. You can only end up here if you were poking around ",
        "the internals and started calling functions you shouldn't be!",
        "\n\n",
        "If you're just looking to start a server component in a JavaScript app,",
        "you can use `lustre.start_server_component`.",
        "\n\n",
        "If you're seeing this error and you think it's a bug. Please open an ",
        "issue over on Github: https://github.com/lustre-labs/lustre/issues/new"
    ].join(""));
};
class Runtime {
    constructor(root, [model, effects], view, update, options){
        this.root = root;
        this.#model = model;
        this.#view = view;
        this.#update = update;
        this.root.addEventListener("context-request", (event)=>{
            // So that we're compatible with other implementations of the proposed
            // protocol, we don't check the event constructor here because other
            // implementations will have defined their own event type.
            if (!(event.context && event.callback)) return;
            if (!this.#contexts.has(event.context)) return;
            event.stopImmediatePropagation();
            const context = this.#contexts.get(event.context);
            if (event.subscribe) {
                const unsubscribe = ()=>{
                    context.subscribers = context.subscribers.filter((subscriber)=>subscriber !== event.callback);
                };
                context.subscribers.push([
                    event.callback,
                    unsubscribe
                ]);
                event.callback(context.value, unsubscribe);
            } else event.callback(context.value);
        });
        const decodeEvent = (event, path, name)=>_cacheMjs.decode(this.#cache, path, name, event);
        const dispatch = (event, data)=>{
            const [cache, result] = _cacheMjs.dispatch(this.#cache, data);
            this.#cache = cache;
            if ((0, _gleamMjs.Result$isOk)(result)) {
                const handler = (0, _gleamMjs.Result$Ok$0)(result);
                if (handler.stop_propagation) event.stopPropagation();
                if (handler.prevent_default) event.preventDefault();
                this.dispatch(handler.message, false);
            }
        };
        this.#reconciler = new (0, _reconcilerFfiMjs.Reconciler)(this.root, decodeEvent, dispatch, options);
        // We want the first render to be synchronous too
        // The initial vdom is whatever we can virtualise from the root node when we
        // mount on to it.
        this.#vdom = (0, _virtualiseFfiMjs.virtualise)(this.root);
        // The initial set of events is empty, since we just virtualised.
        this.#cache = _cacheMjs.new$();
        // // We want the first render to be synchronous and force it immediately.
        // Afterwards, events triggered by virtualisation will dispatch, if any.
        this.#handleEffects(effects);
        this.#render();
    }
    // PUBLIC API ----------------------------------------------------------------
    root = null;
    dispatch(msg, shouldFlush = false) {
        if (this.#shouldQueue) this.#queue.push(msg);
        else {
            const [model, effects] = this.#update(this.#model, msg);
            this.#model = model;
            this.#tick(effects, shouldFlush);
        }
    }
    emit(event, data) {
        const target = this.root.host ?? this.root;
        target.dispatchEvent(new CustomEvent(event, {
            detail: data,
            bubbles: true,
            composed: true
        }));
    }
    // Provide a context value for any child nodes that request it using the given
    // key. If the key already exists, any existing subscribers will be notified
    // of the change. Otherwise, we store the value and wait for any `context-request`
    // events to come in.
    provide(key, value) {
        if (!this.#contexts.has(key)) this.#contexts.set(key, {
            value,
            subscribers: []
        });
        else {
            const context = this.#contexts.get(key);
            // if the new context we provide is equal to the current context,
            // we don't have to notify our subscribers about the change.
            if ((0, _equalsFfiMjs.isEqual)(context.value, value)) return;
            context.value = value;
            for(let i = context.subscribers.length - 1; i >= 0; i--){
                const [subscriber, unsubscribe] = context.subscribers[i];
                // If the subscriber has been garbage collected, we remove it from the
                // list of subscribers.
                if (!subscriber) {
                    context.subscribers.splice(i, 1);
                    continue;
                }
                // Otherwise, we call the subscriber with the new value and the
                // unsubscribe function.
                subscriber(value, unsubscribe);
            }
        }
    }
    // PRIVATE API ---------------------------------------------------------------
    #model;
    #view;
    #update;
    #vdom;
    #cache;
    #reconciler;
    #contexts = new Map();
    #shouldQueue = false;
    #queue = [];
    #beforePaint = (0, _constantsMjs.empty_list);
    #afterPaint = (0, _constantsMjs.empty_list);
    #renderTimer = null;
    #actions = {
        dispatch: (msg)=>this.dispatch(msg),
        emit: (event, data)=>this.emit(event, data),
        select: ()=>{},
        root: ()=>this.root,
        provide: (key, value)=>this.provide(key, value)
    };
    // A `#tick` is where we process effects and trigger any synchronous updates.
    // Once a tick has been processed a render will be scheduled if none is already.
    #tick(effects, shouldFlush = false) {
        this.#handleEffects(effects);
        // queue the next frame if we need to.
        if (!this.#renderTimer) {
            if (shouldFlush) {
                // when rendering synchronously, we still want to delay using a microtask
                // to batch all attribute/property updates.
                this.#renderTimer = "sync";
                queueMicrotask(()=>this.#render());
            } else this.#renderTimer = window.requestAnimationFrame(()=>this.#render());
        }
    }
    // #handleEffects processes all effects, without scheduling a render.
    #handleEffects(effects) {
        // By flipping this on before we process the list of synchronous effects, we
        // make it so that any messages dispatched immediately will be queued up and
        // applied before the next render.
        this.#shouldQueue = true;
        // We step into this loop to process any synchronous effects and batch any
        // deferred ones. When a synchronous effect immediately dispatches a message,
        // we add it to a queue and process another `update` cycle. This continues
        // until there are no more synchronous effects or messages to process.
        while(true){
            // We pass the runtime directly to each effect. It has all the methods
            // of the `Actions` record define in the effect module.
            (0, _listFfiMjs.iterate)(effects.synchronous, (effect)=>effect(this.#actions));
            // Both `before_paint` and `after_paint` are lists of effects that should
            // be deferred until we next perform a render. That means we need to collect
            // them all up in order and save them for later.
            this.#beforePaint = (0, _listFfiMjs.append)(this.#beforePaint, effects.before_paint);
            this.#afterPaint = (0, _listFfiMjs.append)(this.#afterPaint, effects.after_paint);
            // Once we've batched any deferred effects, we check if there are any
            // messages in the queue. If not, we can break out of the loop and continue
            // with the render.
            if (!this.#queue.length) break;
            // This is a destructuring assignment pattern that is mutating both
            // `this.#model` and the argument to this function: `effects`!
            const msg = this.#queue.shift();
            [this.#model, effects] = this.#update(this.#model, msg);
        }
        // Remember to flip this off so subsequent messages trigger another tick.
        this.#shouldQueue = false;
    }
    #render() {
        this.#renderTimer = null;
        const next = this.#view(this.#model);
        const { patch, cache } = (0, _diffMjs.diff)(this.#cache, this.#vdom, next);
        this.#cache = cache;
        this.#vdom = next;
        this.#reconciler.push(patch, _cacheMjs.memos(cache));
        // We have performed a render, the DOM has been updated but the browser has
        // not yet been given the opportunity to paint. We queue a microtask to block
        // the browser from painting until we have processed any effects that need to
        // be run first.
        if ((0, _gleamMjs.List$isNonEmpty)(this.#beforePaint)) {
            const effects = makeEffect(this.#beforePaint);
            this.#beforePaint = (0, _constantsMjs.empty_list);
            // We explicitly queue a microtask instead of synchronously calling the
            // `#tick` function to allow the runtime to process any microtasks queued
            // by synchronous effects first such as promise callbacks.
            queueMicrotask(()=>{
                this.#tick(effects, true);
            });
        }
        // If there are effects to schedule for after the browser has painted, we can
        // request an animation frame and process them then.
        if ((0, _gleamMjs.List$isNonEmpty)(this.#afterPaint)) {
            const effects = makeEffect(this.#afterPaint);
            this.#afterPaint = (0, _constantsMjs.empty_list);
            window.requestAnimationFrame(()=>this.#tick(effects, true));
        }
    }
}
const send = (runtime, message)=>{
    runtime.send(message);
};
//
function makeEffect(synchronous) {
    return {
        synchronous,
        after_paint: (0, _constantsMjs.empty_list),
        before_paint: (0, _constantsMjs.empty_list)
    };
}
const copiedStyleSheets = new WeakMap();
async function adoptStylesheets(shadowRoot) {
    const pendingParentStylesheets = [];
    for (const node of globalThis.document.querySelectorAll("link[rel=stylesheet], style")){
        if (node.sheet) continue;
        pendingParentStylesheets.push(new Promise((resolve, reject)=>{
            node.addEventListener("load", resolve);
            node.addEventListener("error", reject);
        }));
    }
    await Promise.allSettled(pendingParentStylesheets);
    // the element might have been removed while we were waiting.
    if (!shadowRoot.host.isConnected) return [];
    shadowRoot.adoptedStyleSheets = shadowRoot.host.getRootNode().adoptedStyleSheets;
    const pending = [];
    for (const sheet of globalThis.document.styleSheets)try {
        shadowRoot.adoptedStyleSheets.push(sheet);
    } catch  {
        try {
            let copiedSheet = copiedStyleSheets.get(sheet);
            if (!copiedSheet) {
                copiedSheet = new CSSStyleSheet();
                for (const rule of sheet.cssRules)copiedSheet.insertRule(rule.cssText, copiedSheet.cssRules.length);
                copiedStyleSheets.set(sheet, copiedSheet);
            }
            shadowRoot.adoptedStyleSheets.push(copiedSheet);
        } catch  {
            const node = sheet.ownerNode.cloneNode();
            shadowRoot.prepend(node);
            pending.push(node);
        }
    }
    return pending;
}
class ContextRequestEvent extends Event {
    constructor(context, callback, subscribe){
        super("context-request", {
            bubbles: true,
            composed: true
        });
        this.context = context;
        this.callback = callback;
        this.subscribe = subscribe;
    }
}

},{"../../../gleam.mjs":"jNPQG","../../internals/constants.mjs":"gKFR6","../../vdom/diff.mjs":"iOcdA","../../vdom/cache.mjs":"aEh50","../../vdom/reconciler.ffi.mjs":"5QzuP","../../vdom/virtualise.ffi.mjs":"k2cHU","../../internals/equals.ffi.mjs":"2LTPm","../../internals/list.ffi.mjs":"hGVW1","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"iOcdA":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "Diff", ()=>Diff);
parcelHelpers.export(exports, "Diff$Diff", ()=>Diff$Diff);
parcelHelpers.export(exports, "Diff$isDiff", ()=>Diff$isDiff);
parcelHelpers.export(exports, "Diff$Diff$patch", ()=>Diff$Diff$patch);
parcelHelpers.export(exports, "Diff$Diff$0", ()=>Diff$Diff$0);
parcelHelpers.export(exports, "Diff$Diff$cache", ()=>Diff$Diff$cache);
parcelHelpers.export(exports, "Diff$Diff$1", ()=>Diff$Diff$1);
parcelHelpers.export(exports, "diff", ()=>diff);
var _jsonMjs = require("../../../gleam_json/gleam/json.mjs");
var _orderMjs = require("../../../gleam_stdlib/gleam/order.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
var _mutableMapMjs = require("../../lustre/internals/mutable_map.mjs");
var _refMjs = require("../../lustre/internals/ref.mjs");
var _cacheMjs = require("../../lustre/vdom/cache.mjs");
var _patchMjs = require("../../lustre/vdom/patch.mjs");
var _pathMjs = require("../../lustre/vdom/path.mjs");
var _vattrMjs = require("../../lustre/vdom/vattr.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
var _equalsFfiMjs = require("../internals/equals.ffi.mjs");
class Diff extends (0, _gleamMjs.CustomType) {
    constructor(patch, cache){
        super();
        this.patch = patch;
        this.cache = cache;
    }
}
const Diff$Diff = (patch, cache)=>new Diff(patch, cache);
const Diff$isDiff = (value)=>value instanceof Diff;
const Diff$Diff$patch = (value)=>value.patch;
const Diff$Diff$0 = (value)=>value.patch;
const Diff$Diff$cache = (value)=>value.cache;
const Diff$Diff$1 = (value)=>value.cache;
class PartialDiff extends (0, _gleamMjs.CustomType) {
    constructor(patch, cache, events){
        super();
        this.patch = patch;
        this.cache = cache;
        this.events = events;
    }
}
class AttributeChange extends (0, _gleamMjs.CustomType) {
    constructor(added, removed, events){
        super();
        this.added = added;
        this.removed = removed;
        this.events = events;
    }
}
function is_controlled(cache, namespace, tag, path) {
    if (tag === "input" && namespace === "") return _cacheMjs.has_dispatched_events(cache, path);
    else if (tag === "select" && namespace === "") return _cacheMjs.has_dispatched_events(cache, path);
    else if (tag === "textarea" && namespace === "") return _cacheMjs.has_dispatched_events(cache, path);
    else return false;
}
function diff_attributes(loop$controlled, loop$path, loop$events, loop$old, loop$new, loop$added, loop$removed) {
    while(true){
        let controlled = loop$controlled;
        let path = loop$path;
        let events = loop$events;
        let old = loop$old;
        let new$ = loop$new;
        let added = loop$added;
        let removed = loop$removed;
        if (old instanceof (0, _gleamMjs.Empty)) {
            if (new$ instanceof (0, _gleamMjs.Empty)) return new AttributeChange(added, removed, events);
            else {
                let $ = new$.head;
                if ($ instanceof (0, _vattrMjs.Event)) {
                    let next = $;
                    let new$1 = new$.tail;
                    let name = $.name;
                    let handler = $.handler;
                    let events$1 = _cacheMjs.add_event(events, path, name, handler);
                    let added$1 = (0, _gleamMjs.prepend)(next, added);
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = events$1;
                    loop$old = old;
                    loop$new = new$1;
                    loop$added = added$1;
                    loop$removed = removed;
                } else {
                    let next = $;
                    let new$1 = new$.tail;
                    let added$1 = (0, _gleamMjs.prepend)(next, added);
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = events;
                    loop$old = old;
                    loop$new = new$1;
                    loop$added = added$1;
                    loop$removed = removed;
                }
            }
        } else if (new$ instanceof (0, _gleamMjs.Empty)) {
            let $ = old.head;
            if ($ instanceof (0, _vattrMjs.Event)) {
                let prev = $;
                let old$1 = old.tail;
                let name = $.name;
                let events$1 = _cacheMjs.remove_event(events, path, name);
                let removed$1 = (0, _gleamMjs.prepend)(prev, removed);
                loop$controlled = controlled;
                loop$path = path;
                loop$events = events$1;
                loop$old = old$1;
                loop$new = new$;
                loop$added = added;
                loop$removed = removed$1;
            } else {
                let prev = $;
                let old$1 = old.tail;
                let removed$1 = (0, _gleamMjs.prepend)(prev, removed);
                loop$controlled = controlled;
                loop$path = path;
                loop$events = events;
                loop$old = old$1;
                loop$new = new$;
                loop$added = added;
                loop$removed = removed$1;
            }
        } else {
            let prev = old.head;
            let remaining_old = old.tail;
            let next = new$.head;
            let remaining_new = new$.tail;
            let $ = _vattrMjs.compare(prev, next);
            if ($ instanceof (0, _orderMjs.Lt)) {
                if (prev instanceof (0, _vattrMjs.Event)) {
                    let name = prev.name;
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = _cacheMjs.remove_event(events, path, name);
                    loop$old = remaining_old;
                    loop$new = new$;
                    loop$added = added;
                    loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                } else {
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = events;
                    loop$old = remaining_old;
                    loop$new = new$;
                    loop$added = added;
                    loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                }
            } else if ($ instanceof (0, _orderMjs.Eq)) {
                if (prev instanceof (0, _vattrMjs.Attribute)) {
                    if (next instanceof (0, _vattrMjs.Attribute)) {
                        let _block;
                        let $1 = next.name;
                        if ($1 === "value") _block = controlled || prev.value !== next.value;
                        else if ($1 === "checked") _block = controlled || prev.value !== next.value;
                        else if ($1 === "selected") _block = controlled || prev.value !== next.value;
                        else _block = prev.value !== next.value;
                        let has_changes = _block;
                        let _block$1;
                        if (has_changes) _block$1 = (0, _gleamMjs.prepend)(next, added);
                        else _block$1 = added;
                        let added$1 = _block$1;
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = events;
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = added$1;
                        loop$removed = removed;
                    } else if (next instanceof (0, _vattrMjs.Event)) {
                        let name = next.name;
                        let handler = next.handler;
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = _cacheMjs.add_event(events, path, name, handler);
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = (0, _gleamMjs.prepend)(next, added);
                        loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                    } else {
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = events;
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = (0, _gleamMjs.prepend)(next, added);
                        loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                    }
                } else if (prev instanceof (0, _vattrMjs.Property)) {
                    if (next instanceof (0, _vattrMjs.Property)) {
                        let _block;
                        let $1 = next.name;
                        if ($1 === "scrollLeft") _block = true;
                        else if ($1 === "scrollRight") _block = true;
                        else if ($1 === "value") _block = controlled || !(0, _equalsFfiMjs.isEqual)(prev.value, next.value);
                        else if ($1 === "checked") _block = controlled || !(0, _equalsFfiMjs.isEqual)(prev.value, next.value);
                        else if ($1 === "selected") _block = controlled || !(0, _equalsFfiMjs.isEqual)(prev.value, next.value);
                        else _block = !(0, _equalsFfiMjs.isEqual)(prev.value, next.value);
                        let has_changes = _block;
                        let _block$1;
                        if (has_changes) _block$1 = (0, _gleamMjs.prepend)(next, added);
                        else _block$1 = added;
                        let added$1 = _block$1;
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = events;
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = added$1;
                        loop$removed = removed;
                    } else if (next instanceof (0, _vattrMjs.Event)) {
                        let name = next.name;
                        let handler = next.handler;
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = _cacheMjs.add_event(events, path, name, handler);
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = (0, _gleamMjs.prepend)(next, added);
                        loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                    } else {
                        loop$controlled = controlled;
                        loop$path = path;
                        loop$events = events;
                        loop$old = remaining_old;
                        loop$new = remaining_new;
                        loop$added = (0, _gleamMjs.prepend)(next, added);
                        loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                    }
                } else if (next instanceof (0, _vattrMjs.Event)) {
                    let name = next.name;
                    let handler = next.handler;
                    let has_changes = prev.prevent_default.kind !== next.prevent_default.kind || prev.stop_propagation.kind !== next.stop_propagation.kind || prev.debounce !== next.debounce || prev.throttle !== next.throttle;
                    let _block;
                    if (has_changes) _block = (0, _gleamMjs.prepend)(next, added);
                    else _block = added;
                    let added$1 = _block;
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = _cacheMjs.add_event(events, path, name, handler);
                    loop$old = remaining_old;
                    loop$new = remaining_new;
                    loop$added = added$1;
                    loop$removed = removed;
                } else {
                    let name = prev.name;
                    loop$controlled = controlled;
                    loop$path = path;
                    loop$events = _cacheMjs.remove_event(events, path, name);
                    loop$old = remaining_old;
                    loop$new = remaining_new;
                    loop$added = (0, _gleamMjs.prepend)(next, added);
                    loop$removed = (0, _gleamMjs.prepend)(prev, removed);
                }
            } else if (next instanceof (0, _vattrMjs.Event)) {
                let name = next.name;
                let handler = next.handler;
                loop$controlled = controlled;
                loop$path = path;
                loop$events = _cacheMjs.add_event(events, path, name, handler);
                loop$old = old;
                loop$new = remaining_new;
                loop$added = (0, _gleamMjs.prepend)(next, added);
                loop$removed = removed;
            } else {
                loop$controlled = controlled;
                loop$path = path;
                loop$events = events;
                loop$old = old;
                loop$new = remaining_new;
                loop$added = (0, _gleamMjs.prepend)(next, added);
                loop$removed = removed;
            }
        }
    }
}
function do_diff(loop$old, loop$old_keyed, loop$new, loop$new_keyed, loop$moved, loop$moved_offset, loop$removed, loop$node_index, loop$patch_index, loop$changes, loop$children, loop$path, loop$cache, loop$events) {
    while(true){
        let old = loop$old;
        let old_keyed = loop$old_keyed;
        let new$ = loop$new;
        let new_keyed = loop$new_keyed;
        let moved = loop$moved;
        let moved_offset = loop$moved_offset;
        let removed = loop$removed;
        let node_index = loop$node_index;
        let patch_index = loop$patch_index;
        let changes = loop$changes;
        let children = loop$children;
        let path = loop$path;
        let cache = loop$cache;
        let events = loop$events;
        if (old instanceof (0, _gleamMjs.Empty)) {
            if (new$ instanceof (0, _gleamMjs.Empty)) {
                let patch = new (0, _patchMjs.Patch)(patch_index, removed, changes, children);
                return new PartialDiff(patch, cache, events);
            } else {
                let $ = _cacheMjs.add_children(cache, events, path, node_index, new$);
                let cache$1;
                let events$1;
                cache$1 = $[0];
                events$1 = $[1];
                let insert = _patchMjs.insert(new$, node_index - moved_offset);
                let changes$1 = (0, _gleamMjs.prepend)(insert, changes);
                let patch = new (0, _patchMjs.Patch)(patch_index, removed, changes$1, children);
                return new PartialDiff(patch, cache$1, events$1);
            }
        } else if (new$ instanceof (0, _gleamMjs.Empty)) {
            let prev = old.head;
            let old$1 = old.tail;
            let _block;
            let $ = prev.key === "" || !_mutableMapMjs.has_key(moved, prev.key);
            if ($) _block = removed + 1;
            else _block = removed;
            let removed$1 = _block;
            let events$1 = _cacheMjs.remove_child(cache, events, path, node_index, prev);
            loop$old = old$1;
            loop$old_keyed = old_keyed;
            loop$new = new$;
            loop$new_keyed = new_keyed;
            loop$moved = moved;
            loop$moved_offset = moved_offset;
            loop$removed = removed$1;
            loop$node_index = node_index;
            loop$patch_index = patch_index;
            loop$changes = changes;
            loop$children = children;
            loop$path = path;
            loop$cache = cache;
            loop$events = events$1;
        } else {
            let prev = old.head;
            let next = new$.head;
            if (prev.key !== next.key) {
                let old_remaining = old.tail;
                let new_remaining = new$.tail;
                let next_did_exist = _mutableMapMjs.has_key(old_keyed, next.key);
                let prev_does_exist = _mutableMapMjs.has_key(new_keyed, prev.key);
                if (prev_does_exist) {
                    if (next_did_exist) {
                        let $ = _mutableMapMjs.has_key(moved, prev.key);
                        if ($) {
                            loop$old = old_remaining;
                            loop$old_keyed = old_keyed;
                            loop$new = new$;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset - 1;
                            loop$removed = removed;
                            loop$node_index = node_index;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache;
                            loop$events = events;
                        } else {
                            let match = _mutableMapMjs.unsafe_get(old_keyed, next.key);
                            let before = node_index - moved_offset;
                            let changes$1 = (0, _gleamMjs.prepend)(_patchMjs.move(next.key, before), changes);
                            let moved$1 = _mutableMapMjs.insert(moved, next.key, undefined);
                            loop$old = (0, _gleamMjs.prepend)(match, old);
                            loop$old_keyed = old_keyed;
                            loop$new = new$;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved$1;
                            loop$moved_offset = moved_offset + 1;
                            loop$removed = removed;
                            loop$node_index = node_index;
                            loop$patch_index = patch_index;
                            loop$changes = changes$1;
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache;
                            loop$events = events;
                        }
                    } else {
                        let before = node_index - moved_offset;
                        let $ = _cacheMjs.add_child(cache, events, path, node_index, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $[0];
                        events$1 = $[1];
                        let insert = _patchMjs.insert((0, _gleamMjs.toList)([
                            next
                        ]), before);
                        let changes$1 = (0, _gleamMjs.prepend)(insert, changes);
                        loop$old = old;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset + 1;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = changes$1;
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else if (next_did_exist) {
                    let index = node_index - moved_offset;
                    let changes$1 = (0, _gleamMjs.prepend)(_patchMjs.remove(index), changes);
                    let events$1 = _cacheMjs.remove_child(cache, events, path, node_index, prev);
                    loop$old = old_remaining;
                    loop$old_keyed = old_keyed;
                    loop$new = new$;
                    loop$new_keyed = new_keyed;
                    loop$moved = moved;
                    loop$moved_offset = moved_offset - 1;
                    loop$removed = removed;
                    loop$node_index = node_index;
                    loop$patch_index = patch_index;
                    loop$changes = changes$1;
                    loop$children = children;
                    loop$path = path;
                    loop$cache = cache;
                    loop$events = events$1;
                } else {
                    let change = _patchMjs.replace(node_index - moved_offset, next);
                    let $ = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                    let cache$1;
                    let events$1;
                    cache$1 = $[0];
                    events$1 = $[1];
                    loop$old = old_remaining;
                    loop$old_keyed = old_keyed;
                    loop$new = new_remaining;
                    loop$new_keyed = new_keyed;
                    loop$moved = moved;
                    loop$moved_offset = moved_offset;
                    loop$removed = removed;
                    loop$node_index = node_index + 1;
                    loop$patch_index = patch_index;
                    loop$changes = (0, _gleamMjs.prepend)(change, changes);
                    loop$children = children;
                    loop$path = path;
                    loop$cache = cache$1;
                    loop$events = events$1;
                }
            } else {
                let $ = old.head;
                if ($ instanceof (0, _vnodeMjs.Fragment)) {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.Fragment)) {
                        let prev = $;
                        let old$1 = old.tail;
                        let next = $1;
                        let new$1 = new$.tail;
                        let $2 = do_diff(prev.children, prev.keyed_children, next.children, next.keyed_children, _mutableMapMjs.new$(), 0, 0, 0, node_index, _constantsMjs.empty_list, _constantsMjs.empty_list, _pathMjs.add(path, node_index, next.key), cache, events);
                        let patch;
                        let cache$1;
                        let events$1;
                        patch = $2.patch;
                        cache$1 = $2.cache;
                        events$1 = $2.events;
                        let _block;
                        let $3 = patch.changes;
                        if ($3 instanceof (0, _gleamMjs.Empty)) {
                            let $4 = patch.children;
                            if ($4 instanceof (0, _gleamMjs.Empty)) {
                                let $5 = patch.removed;
                                if ($5 === 0) _block = children;
                                else _block = (0, _gleamMjs.prepend)(patch, children);
                            } else _block = (0, _gleamMjs.prepend)(patch, children);
                        } else _block = (0, _gleamMjs.prepend)(patch, children);
                        let children$1 = _block;
                        loop$old = old$1;
                        loop$old_keyed = old_keyed;
                        loop$new = new$1;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = changes;
                        loop$children = children$1;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else if ($ instanceof (0, _vnodeMjs.Element)) {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.Element)) {
                        let prev = $;
                        let next = $1;
                        if (prev.namespace === next.namespace && prev.tag === next.tag) {
                            let old$1 = old.tail;
                            let new$1 = new$.tail;
                            let child_path = _pathMjs.add(path, node_index, next.key);
                            let controlled = is_controlled(cache, next.namespace, next.tag, child_path);
                            let $2 = diff_attributes(controlled, child_path, events, prev.attributes, next.attributes, _constantsMjs.empty_list, _constantsMjs.empty_list);
                            let added_attrs;
                            let removed_attrs;
                            let events$1;
                            added_attrs = $2.added;
                            removed_attrs = $2.removed;
                            events$1 = $2.events;
                            let _block;
                            if (added_attrs instanceof (0, _gleamMjs.Empty) && removed_attrs instanceof (0, _gleamMjs.Empty)) _block = _constantsMjs.empty_list;
                            else _block = (0, _gleamMjs.toList)([
                                _patchMjs.update(added_attrs, removed_attrs)
                            ]);
                            let initial_child_changes = _block;
                            let $3 = do_diff(prev.children, prev.keyed_children, next.children, next.keyed_children, _mutableMapMjs.new$(), 0, 0, 0, node_index, initial_child_changes, _constantsMjs.empty_list, child_path, cache, events$1);
                            let patch;
                            let cache$1;
                            let events$2;
                            patch = $3.patch;
                            cache$1 = $3.cache;
                            events$2 = $3.events;
                            let _block$1;
                            let $4 = patch.changes;
                            if ($4 instanceof (0, _gleamMjs.Empty)) {
                                let $5 = patch.children;
                                if ($5 instanceof (0, _gleamMjs.Empty)) {
                                    let $6 = patch.removed;
                                    if ($6 === 0) _block$1 = children;
                                    else _block$1 = (0, _gleamMjs.prepend)(patch, children);
                                } else _block$1 = (0, _gleamMjs.prepend)(patch, children);
                            } else _block$1 = (0, _gleamMjs.prepend)(patch, children);
                            let children$1 = _block$1;
                            loop$old = old$1;
                            loop$old_keyed = old_keyed;
                            loop$new = new$1;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index + 1;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = children$1;
                            loop$path = path;
                            loop$cache = cache$1;
                            loop$events = events$2;
                        } else {
                            let prev = $;
                            let old_remaining = old.tail;
                            let next = $1;
                            let new_remaining = new$.tail;
                            let change = _patchMjs.replace(node_index - moved_offset, next);
                            let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                            let cache$1;
                            let events$1;
                            cache$1 = $2[0];
                            events$1 = $2[1];
                            loop$old = old_remaining;
                            loop$old_keyed = old_keyed;
                            loop$new = new_remaining;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index + 1;
                            loop$patch_index = patch_index;
                            loop$changes = (0, _gleamMjs.prepend)(change, changes);
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache$1;
                            loop$events = events$1;
                        }
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else if ($ instanceof (0, _vnodeMjs.Text)) {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.Text)) {
                        let prev = $;
                        let next = $1;
                        if (prev.content === next.content) {
                            let old$1 = old.tail;
                            let new$1 = new$.tail;
                            loop$old = old$1;
                            loop$old_keyed = old_keyed;
                            loop$new = new$1;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index + 1;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache;
                            loop$events = events;
                        } else {
                            let old$1 = old.tail;
                            let next = $1;
                            let new$1 = new$.tail;
                            let child = _patchMjs.new$(node_index, 0, (0, _gleamMjs.toList)([
                                _patchMjs.replace_text(next.content)
                            ]), _constantsMjs.empty_list);
                            loop$old = old$1;
                            loop$old_keyed = old_keyed;
                            loop$new = new$1;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index + 1;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = (0, _gleamMjs.prepend)(child, children);
                            loop$path = path;
                            loop$cache = cache;
                            loop$events = events;
                        }
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else if ($ instanceof (0, _vnodeMjs.UnsafeInnerHtml)) {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.UnsafeInnerHtml)) {
                        let prev = $;
                        let old$1 = old.tail;
                        let next = $1;
                        let new$1 = new$.tail;
                        let child_path = _pathMjs.add(path, node_index, next.key);
                        let $2 = diff_attributes(false, child_path, events, prev.attributes, next.attributes, _constantsMjs.empty_list, _constantsMjs.empty_list);
                        let added_attrs;
                        let removed_attrs;
                        let events$1;
                        added_attrs = $2.added;
                        removed_attrs = $2.removed;
                        events$1 = $2.events;
                        let _block;
                        if (added_attrs instanceof (0, _gleamMjs.Empty) && removed_attrs instanceof (0, _gleamMjs.Empty)) _block = _constantsMjs.empty_list;
                        else _block = (0, _gleamMjs.toList)([
                            _patchMjs.update(added_attrs, removed_attrs)
                        ]);
                        let child_changes = _block;
                        let _block$1;
                        let $3 = prev.inner_html === next.inner_html;
                        if ($3) _block$1 = child_changes;
                        else _block$1 = (0, _gleamMjs.prepend)(_patchMjs.replace_inner_html(next.inner_html), child_changes);
                        let child_changes$1 = _block$1;
                        let _block$2;
                        if (child_changes$1 instanceof (0, _gleamMjs.Empty)) _block$2 = children;
                        else _block$2 = (0, _gleamMjs.prepend)(_patchMjs.new$(node_index, 0, child_changes$1, (0, _gleamMjs.toList)([])), children);
                        let children$1 = _block$2;
                        loop$old = old$1;
                        loop$old_keyed = old_keyed;
                        loop$new = new$1;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = changes;
                        loop$children = children$1;
                        loop$path = path;
                        loop$cache = cache;
                        loop$events = events$1;
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else if ($ instanceof (0, _vnodeMjs.Map)) {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.Map)) {
                        let prev = $;
                        let old$1 = old.tail;
                        let next = $1;
                        let new$1 = new$.tail;
                        let child_path = _pathMjs.add(path, node_index, next.key);
                        let child_key = _pathMjs.child(child_path);
                        let $2 = do_diff((0, _gleamMjs.prepend)(prev.child, _constantsMjs.empty_list), _mutableMapMjs.new$(), (0, _gleamMjs.prepend)(next.child, _constantsMjs.empty_list), _mutableMapMjs.new$(), _mutableMapMjs.new$(), 0, 0, 0, node_index, _constantsMjs.empty_list, _constantsMjs.empty_list, _pathMjs.subtree(child_path), cache, _cacheMjs.get_subtree(events, child_key, prev.mapper));
                        let patch;
                        let cache$1;
                        let child_events;
                        patch = $2.patch;
                        cache$1 = $2.cache;
                        child_events = $2.events;
                        let events$1 = _cacheMjs.update_subtree(events, child_key, next.mapper, child_events);
                        let _block;
                        let $3 = patch.changes;
                        if ($3 instanceof (0, _gleamMjs.Empty)) {
                            let $4 = patch.children;
                            if ($4 instanceof (0, _gleamMjs.Empty)) {
                                let $5 = patch.removed;
                                if ($5 === 0) _block = children;
                                else _block = (0, _gleamMjs.prepend)(patch, children);
                            } else _block = (0, _gleamMjs.prepend)(patch, children);
                        } else _block = (0, _gleamMjs.prepend)(patch, children);
                        let children$1 = _block;
                        loop$old = old$1;
                        loop$old_keyed = old_keyed;
                        loop$new = new$1;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = changes;
                        loop$children = children$1;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                } else {
                    let $1 = new$.head;
                    if ($1 instanceof (0, _vnodeMjs.Memo)) {
                        let prev = $;
                        let old$1 = old.tail;
                        let next = $1;
                        let new$1 = new$.tail;
                        let $2 = _refMjs.equal_lists(prev.dependencies, next.dependencies);
                        if ($2) {
                            let cache$1 = _cacheMjs.keep_memo(cache, prev.view, next.view);
                            loop$old = old$1;
                            loop$old_keyed = old_keyed;
                            loop$new = new$1;
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index + 1;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache$1;
                            loop$events = events;
                        } else {
                            let prev_node = _cacheMjs.get_old_memo(cache, prev.view, prev.view);
                            let next_node = next.view();
                            let cache$1 = _cacheMjs.add_memo(cache, next.view, next_node);
                            loop$old = (0, _gleamMjs.prepend)(prev_node, old$1);
                            loop$old_keyed = old_keyed;
                            loop$new = (0, _gleamMjs.prepend)(next_node, new$1);
                            loop$new_keyed = new_keyed;
                            loop$moved = moved;
                            loop$moved_offset = moved_offset;
                            loop$removed = removed;
                            loop$node_index = node_index;
                            loop$patch_index = patch_index;
                            loop$changes = changes;
                            loop$children = children;
                            loop$path = path;
                            loop$cache = cache$1;
                            loop$events = events;
                        }
                    } else {
                        let prev = $;
                        let old_remaining = old.tail;
                        let next = $1;
                        let new_remaining = new$.tail;
                        let change = _patchMjs.replace(node_index - moved_offset, next);
                        let $2 = _cacheMjs.replace_child(cache, events, path, node_index, prev, next);
                        let cache$1;
                        let events$1;
                        cache$1 = $2[0];
                        events$1 = $2[1];
                        loop$old = old_remaining;
                        loop$old_keyed = old_keyed;
                        loop$new = new_remaining;
                        loop$new_keyed = new_keyed;
                        loop$moved = moved;
                        loop$moved_offset = moved_offset;
                        loop$removed = removed;
                        loop$node_index = node_index + 1;
                        loop$patch_index = patch_index;
                        loop$changes = (0, _gleamMjs.prepend)(change, changes);
                        loop$children = children;
                        loop$path = path;
                        loop$cache = cache$1;
                        loop$events = events$1;
                    }
                }
            }
        }
    }
}
function diff(cache, old, new$) {
    let cache$1 = _cacheMjs.tick(cache);
    let $ = do_diff((0, _gleamMjs.prepend)(old, _constantsMjs.empty_list), _mutableMapMjs.new$(), (0, _gleamMjs.prepend)(new$, _constantsMjs.empty_list), _mutableMapMjs.new$(), _mutableMapMjs.new$(), 0, 0, 0, 0, _constantsMjs.empty_list, _constantsMjs.empty_list, _pathMjs.root, cache$1, _cacheMjs.events(cache$1));
    let patch;
    let cache$2;
    let events;
    patch = $.patch;
    cache$2 = $.cache;
    events = $.events;
    return new Diff(patch, _cacheMjs.update_events(cache$2, events));
}

},{"../../../gleam_json/gleam/json.mjs":"8Pq32","../../../gleam_stdlib/gleam/order.mjs":"eYj92","../../gleam.mjs":"jNPQG","../../lustre/internals/constants.mjs":"gKFR6","../../lustre/internals/mutable_map.mjs":"6NvMa","../../lustre/internals/ref.mjs":"gnct2","../../lustre/vdom/cache.mjs":"aEh50","../../lustre/vdom/patch.mjs":"31vMv","../../lustre/vdom/path.mjs":"351yX","../../lustre/vdom/vattr.mjs":"jrrcC","../../lustre/vdom/vnode.mjs":"j2vnp","../internals/equals.ffi.mjs":"2LTPm","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"2LTPm":[function(require,module,exports,__globalThis) {
// This isEqual implementation has to support JSON literals, i.e. values that
// can be produced by using the gleam/json module.
// It is a highly specialised version of https://github.com/planttheidea/fast-equals.
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isEqual", ()=>isEqual);
const isEqual = (a, b)=>{
    if (a === b) return true;
    if (a == null || b == null) return false;
    const type = typeof a;
    if (type !== typeof b) return false;
    // we do not support NaN, and both values being equal has already
    // been handled above.
    if (type !== "object") return false;
    const ctor = a.constructor;
    if (ctor !== b.constructor) return false;
    if (Array.isArray(a)) return areArraysEqual(a, b);
    return areObjectsEqual(a, b);
};
const areArraysEqual = (a, b)=>{
    let index = a.length;
    if (index !== b.length) return false;
    while(index--){
        if (!isEqual(a[index], b[index])) return false;
    }
    return true;
};
const areObjectsEqual = (a, b)=>{
    const properties = Object.keys(a);
    let index = properties.length;
    if (Object.keys(b).length !== index) return false;
    while(index--){
        const property = properties[index];
        if (!Object.hasOwn(b, property)) return false;
        if (!isEqual(a[property], b[property])) return false;
    }
    return true;
};

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5QzuP":[function(require,module,exports,__globalThis) {
// IMPORTS ---------------------------------------------------------------------
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "isLustreNode", ()=>isLustreNode);
parcelHelpers.export(exports, "insertMetadataChild", ()=>insertMetadataChild);
// RECONCILER ------------------------------------------------------------------
parcelHelpers.export(exports, "Reconciler", ()=>Reconciler);
var _houdiniMjs = require("../../../houdini/houdini.mjs");
var _vnodeMjs = require("./vnode.mjs");
var _vattrMjs = require("./vattr.mjs");
var _patchMjs = require("./patch.mjs");
var _pathMjs = require("./path.mjs");
var _listFfiMjs = require("../internals/list.ffi.mjs");
var _constantsFfiMjs = require("../internals/constants.ffi.mjs");
//
// DOM API ---------------------------------------------------------------------
// We do this for 2 reasons:
//
// - Improve code size by only spelling out the property names in one place
//
// - Avoid megamorphic call sites by avoiding direct DOM accesses in the
//   reconciler main functions.
//
// We could directly store references to the Node.protoype functions too and
// avoid chasing the prototype chains, however that would break many DOM crimes
// we want to do: for example for the portal or transition components.
const setTimeout = globalThis.setTimeout;
const clearTimeout = globalThis.clearTimeout;
const createElementNS = (ns, name)=>globalThis.document.createElementNS(ns, name);
const createTextNode = (data)=>globalThis.document.createTextNode(data);
const createComment = (data)=>globalThis.document.createComment(data);
const createDocumentFragment = ()=>globalThis.document.createDocumentFragment();
const insertBefore = (parent, node, reference)=>parent.insertBefore(node, reference);
const moveBefore = (0, _constantsFfiMjs.SUPPORTS_MOVE_BEFORE) ? (parent, node, reference)=>parent.moveBefore(node, reference) : insertBefore;
const removeChild = (parent, child)=>parent.removeChild(child);
const getAttribute = (node, name)=>node.getAttribute(name);
const setAttribute = (node, name, value)=>node.setAttribute(name, value);
const removeAttribute = (node, name)=>node.removeAttribute(name);
const addEventListener = (node, name, handler, options)=>node.addEventListener(name, handler, options);
const removeEventListener = (node, name, handler)=>node.removeEventListener(name, handler);
const setInnerHtml = (node, innerHtml)=>node.innerHTML = innerHtml;
const setData = (node, data)=>node.data = data;
// METADATA / STATEFUL TREE ----------------------------------------------------
// We store some additional data for every node that we create.
// We store that "metadata" using a symbol on each DOM node.
const meta = Symbol("lustre");
class MetadataNode {
    constructor(kind, parent, node, key){
        this.kind = kind;
        // store the key of the element to be able to reconstruct the path
        // once an event gets dispatched.
        this.key = key;
        // parent and children point to the _metadata_ nodes.
        this.parent = parent;
        this.children = [];
        // a reference back to the "real" DOM node.
        this.node = node;
        // in "debug" mode or after virtualisation, fragments also have an "end" marker.
        // we need to move and modify that end marker with the fragment if it exists.
        this.endNode = null;
        // data for the event handlers and attached throttlers and debouncers.
        this.handlers = new Map();
        this.throttles = new Map();
        this.debouncers = new Map();
    }
    get isVirtual() {
        return this.kind === (0, _vnodeMjs.fragment_kind) || this.kind === (0, _vnodeMjs.map_kind);
    }
    get parentNode() {
        return this.isVirtual ? this.node.parentNode : this.node;
    }
}
const isLustreNode = (node)=>node[meta] instanceof MetadataNode;
const insertMetadataChild = (kind, parent, node, index, key)=>{
    const child = new MetadataNode(kind, parent, node, key);
    node[meta] = child;
    parent?.children.splice(index, 0, child);
    return child;
};
const getPath = (node)=>{
    let path = "";
    for(let current = node[meta]; current.parent; current = current.parent){
        // Map nodes use a different separator to mark isolated event subtrees.
        // This allows the cache to split paths and look up handlers in the correct
        // subtree, keeping event handlers stable when parent Map nodes update.
        const separator = current.parent && current.parent.kind === (0, _vnodeMjs.map_kind) ? (0, _pathMjs.separator_subtree) : (0, _pathMjs.separator_element);
        if (current.key) path = `${separator}${current.key}${path}`;
        else {
            const index = current.parent.children.indexOf(current);
            path = `${separator}${index}${path}`;
        }
    }
    // remove the leading separator.
    return path.slice(1);
};
class Reconciler {
    #root = null;
    #decodeEvent;
    #dispatch;
    #debug = false;
    constructor(root, decodeEvent, dispatch, { debug = false } = {}){
        this.#root = root;
        this.#decodeEvent = decodeEvent;
        this.#dispatch = dispatch;
        this.#debug = debug;
    }
    mount(vdom) {
        insertMetadataChild((0, _vnodeMjs.element_kind), null, this.#root, 0, null);
        this.#insertChild(this.#root, null, this.#root[meta], 0, vdom);
    }
    push(patch, memos = null) {
        this.#memos = memos;
        this.#stack.push({
            node: this.#root[meta],
            patch: patch
        });
        this.#reconcile();
    }
    // PATCHING ------------------------------------------------------------------
    #memos;
    #stack = [];
    #reconcile() {
        const stack = this.#stack;
        while(stack.length){
            const { node, patch } = stack.pop();
            const { children: childNodes } = node;
            const { changes, removed, children: childPatches } = patch;
            (0, _listFfiMjs.iterate)(changes, (change)=>this.#patch(node, change));
            if (removed) this.#removeChildren(node, childNodes.length - removed, removed);
            (0, _listFfiMjs.iterate)(childPatches, (childPatch)=>{
                const child = childNodes[childPatch.index | 0];
                this.#stack.push({
                    node: child,
                    patch: childPatch
                });
            });
        }
    }
    #patch(node, change) {
        switch(change.kind){
            case 0, _patchMjs.replace_text_kind:
                this.#replaceText(node, change);
                break;
            case 0, _patchMjs.replace_inner_html_kind:
                this.#replaceInnerHtml(node, change);
                break;
            case 0, _patchMjs.update_kind:
                this.#update(node, change);
                break;
            case 0, _patchMjs.move_kind:
                this.#move(node, change);
                break;
            case 0, _patchMjs.remove_kind:
                this.#remove(node, change);
                break;
            case 0, _patchMjs.replace_kind:
                this.#replace(node, change);
                break;
            case 0, _patchMjs.insert_kind:
                this.#insert(node, change);
                break;
        }
    }
    // CHANGES -------------------------------------------------------------------
    #insert(parent, { children, before }) {
        const fragment = createDocumentFragment();
        const beforeEl = this.#getReference(parent, before);
        this.#insertChildren(fragment, null, parent, before | 0, children);
        insertBefore(parent.parentNode, fragment, beforeEl);
    }
    #replace(parent, { index, with: child }) {
        this.#removeChildren(parent, index | 0, 1);
        const beforeEl = this.#getReference(parent, index);
        this.#insertChild(parent.parentNode, beforeEl, parent, index | 0, child);
    }
    #getReference(node, index) {
        index = index | 0;
        const { children } = node;
        const childCount = children.length;
        if (index < childCount) return children[index].node;
        if (node.endNode) return node.endNode;
        if (!node.isVirtual) return null;
        // unwrap the last child as long as we point to a fragment.
        // otherwise, the fragments next sibling would be the first child of the
        // fragment, not the first element after it.
        while(node.isVirtual && node.children.length){
            if (node.endNode) return node.endNode.nextSibling;
            node = node.children[node.children.length - 1];
        }
        return node.node.nextSibling;
    }
    #move(parent, { key, before }) {
        before = before | 0;
        const { children, parentNode } = parent;
        // unlike insert, we always have to have the before element here!
        const beforeEl = children[before].node;
        let prev = children[before];
        // we only move items to earlier positions, so we can start searching at before + 1.
        for(let i = before + 1; i < children.length; ++i){
            const next = children[i];
            // we shift items from before to the key over one-by-one, to make room
            // for the moved element at children[before].
            children[i] = prev;
            prev = next;
            if (next.key === key) {
                children[before] = next;
                break;
            }
        }
        // prev now is the same as `next` inside the loop, and points to the element
        // we found that matches the key! all that's left is to move it before `beforeEl`.
        this.#moveChild(parentNode, prev, beforeEl);
    }
    #moveChildren(domParent, children, beforeEl) {
        for(let i = 0; i < children.length; ++i)this.#moveChild(domParent, children[i], beforeEl);
    }
    #moveChild(domParent, child, beforeEl) {
        moveBefore(domParent, child.node, beforeEl);
        // child might be a fragment, in which case we need do move all its child dom nodes too
        if (child.isVirtual) this.#moveChildren(domParent, child.children, beforeEl);
        // if "endNode" is set, that node is also a sibling node that we need to move with the children
        if (child.endNode) moveBefore(domParent, child.endNode, beforeEl);
    }
    #remove(parent, { index }) {
        this.#removeChildren(parent, index, 1);
    }
    #removeChildren(parent, index, count) {
        const { children, parentNode } = parent;
        const deleted = children.splice(index, count);
        for(let i = 0; i < deleted.length; ++i){
            const child = deleted[i];
            const { node, endNode, isVirtual, children: nestedChildren } = child;
            removeChild(parentNode, node);
            if (endNode) removeChild(parentNode, endNode);
            this.#removeDebouncers(child);
            if (isVirtual) deleted.push(...nestedChildren);
        }
    }
    #removeDebouncers(node) {
        const { debouncers, children } = node;
        for (const { timeout } of debouncers.values())if (timeout) clearTimeout(timeout);
        debouncers.clear();
        (0, _listFfiMjs.iterate)(children, (child)=>this.#removeDebouncers(child));
    }
    #update({ node, handlers, throttles, debouncers }, { added, removed }) {
        (0, _listFfiMjs.iterate)(removed, ({ name })=>{
            if (handlers.delete(name)) {
                removeEventListener(node, name, handleEvent);
                this.#updateDebounceThrottle(throttles, name, 0);
                this.#updateDebounceThrottle(debouncers, name, 0);
            } else {
                removeAttribute(node, name);
                SYNCED_ATTRIBUTES[name]?.removed?.(node, name);
            }
        });
        (0, _listFfiMjs.iterate)(added, (attribute)=>this.#createAttribute(node, attribute));
    }
    #replaceText({ node }, { content }) {
        setData(node, content ?? "");
    }
    #replaceInnerHtml({ node }, { inner_html }) {
        setInnerHtml(node, inner_html ?? "");
    }
    // INSERT --------------------------------------------------------------------
    #insertChildren(domParent, beforeEl, metaParent, index, children) {
        (0, _listFfiMjs.iterate)(children, (child)=>this.#insertChild(domParent, beforeEl, metaParent, index++, child));
    }
    #insertChild(domParent, beforeEl, metaParent, index, vnode) {
        switch(vnode.kind){
            case 0, _vnodeMjs.element_kind:
                {
                    const node = this.#createElement(metaParent, index, vnode);
                    this.#insertChildren(node, null, node[meta], 0, vnode.children);
                    insertBefore(domParent, node, beforeEl);
                    break;
                }
            case 0, _vnodeMjs.text_kind:
                {
                    const node = this.#createTextNode(metaParent, index, vnode);
                    insertBefore(domParent, node, beforeEl);
                    break;
                }
            case 0, _vnodeMjs.fragment_kind:
                {
                    const marker = "lustre:fragment";
                    const head = this.#createHead(marker, metaParent, index, vnode);
                    insertBefore(domParent, head, beforeEl);
                    this.#insertChildren(domParent, beforeEl, head[meta], 0, vnode.children);
                    if (this.#debug) {
                        head[meta].endNode = createComment(` /${marker} `);
                        insertBefore(domParent, head[meta].endNode, beforeEl);
                    }
                    break;
                }
            case 0, _vnodeMjs.unsafe_inner_html_kind:
                {
                    const node = this.#createElement(metaParent, index, vnode);
                    this.#replaceInnerHtml({
                        node
                    }, vnode);
                    insertBefore(domParent, node, beforeEl);
                    break;
                }
            case 0, _vnodeMjs.map_kind:
                {
                    // Map nodes are virtual like fragments; this allows us to track
                    // subtree boundaries in the real DOM and construct event paths accordingly.
                    const head = this.#createHead("lustre:map", metaParent, index, vnode);
                    insertBefore(domParent, head, beforeEl);
                    this.#insertChild(domParent, beforeEl, head[meta], 0, vnode.child);
                    break;
                }
            case 0, _vnodeMjs.memo_kind:
                {
                    // NOTE: we do not get memo nodes when running as a server component!
                    // Memo nodes are always transparent - they don't create DOM nodes even in debug mode.
                    const child = this.#memos?.get(vnode.view) ?? vnode.view();
                    this.#insertChild(domParent, beforeEl, metaParent, index, child);
                    break;
                }
        }
    }
    #createElement(parent, index, { kind, key, tag, namespace, attributes }) {
        const node = createElementNS(namespace || (0, _constantsFfiMjs.NAMESPACE_HTML), tag);
        insertMetadataChild(kind, parent, node, index, key);
        if (this.#debug && key) setAttribute(node, "data-lustre-key", key);
        (0, _listFfiMjs.iterate)(attributes, (attribute)=>this.#createAttribute(node, attribute));
        return node;
    }
    #createTextNode(parent, index, { kind, key, content }) {
        const node = createTextNode(content ?? "");
        insertMetadataChild(kind, parent, node, index, key);
        return node;
    }
    #createHead(marker, parent, index, { kind, key }) {
        const node = this.#debug ? createComment(markerComment(marker, key)) : createTextNode("");
        insertMetadataChild(kind, parent, node, index, key);
        return node;
    }
    #createAttribute(node, attribute) {
        const { debouncers, handlers, throttles } = node[meta];
        const { kind, name, value, prevent_default: prevent, debounce: debounceDelay, throttle: throttleDelay } = attribute;
        switch(kind){
            case 0, _vattrMjs.attribute_kind:
                {
                    const valueOrDefault = value ?? "";
                    if (name === "virtual:defaultValue") {
                        node.defaultValue = valueOrDefault;
                        return;
                    } else if (name === "virtual:defaultChecked") {
                        node.defaultChecked = true;
                        return;
                    } else if (name === "virtual:defaultSelected") {
                        node.defaultSelected = true;
                        return;
                    }
                    if (valueOrDefault !== getAttribute(node, name)) setAttribute(node, name, valueOrDefault);
                    SYNCED_ATTRIBUTES[name]?.added?.(node, valueOrDefault);
                    break;
                }
            case 0, _vattrMjs.property_kind:
                node[name] = value;
                break;
            case 0, _vattrMjs.event_kind:
                {
                    if (handlers.has(name)) // we re-attach an event listener on every change in case we need
                    // to change the options we pass.
                    removeEventListener(node, name, handleEvent);
                    const passive = prevent.kind === (0, _vattrMjs.never_kind);
                    addEventListener(node, name, handleEvent, {
                        passive
                    });
                    this.#updateDebounceThrottle(throttles, name, throttleDelay);
                    this.#updateDebounceThrottle(debouncers, name, debounceDelay);
                    handlers.set(name, (event)=>this.#handleEvent(attribute, event));
                    break;
                }
        }
    }
    #updateDebounceThrottle(map, name, delay) {
        const debounceOrThrottle = map.get(name);
        if (delay > 0) {
            if (debounceOrThrottle) debounceOrThrottle.delay = delay;
            else map.set(name, {
                delay
            });
        } else if (debounceOrThrottle) {
            const { timeout } = debounceOrThrottle;
            if (timeout) clearTimeout(timeout);
            map.delete(name);
        }
    }
    #handleEvent(attribute, event) {
        const { currentTarget, type } = event;
        const { debouncers, throttles } = currentTarget[meta];
        const path = getPath(currentTarget);
        const { prevent_default: prevent, stop_propagation: stop, include } = attribute;
        if (prevent.kind === (0, _vattrMjs.always_kind)) event.preventDefault();
        if (stop.kind === (0, _vattrMjs.always_kind)) event.stopPropagation();
        if (type === "submit") {
            event.detail ??= {};
            event.detail.formData = [
                ...new FormData(event.target, event.submitter).entries()
            ];
        }
        const data = this.#decodeEvent(event, path, type, include);
        const throttle = throttles.get(type);
        if (throttle) {
            const now = Date.now();
            const last = throttle.last || 0;
            if (now > last + throttle.delay) {
                throttle.last = now;
                throttle.lastEvent = event;
                this.#dispatch(event, data);
            }
        }
        const debounce = debouncers.get(type);
        if (debounce) {
            clearTimeout(debounce.timeout);
            debounce.timeout = setTimeout(()=>{
                if (event === throttles.get(type)?.lastEvent) return;
                this.#dispatch(event, data);
            }, debounce.delay);
        }
        if (!throttle && !debounce) this.#dispatch(event, data);
    }
}
// UTILS -----------------------------------------------------------------------
const markerComment = (marker, key)=>{
    if (key) return ` ${marker} key="${(0, _houdiniMjs.escape)(key)}" `;
    else return ` ${marker} `;
};
// EVENTS ----------------------------------------------------------------------
/** Stable references to an element's event handler are necessary if you ever want
 *  to actually remove them. To achieve that we define this shell `handleEvent`
 *  function that just delegates to an actual event handler stored on the node
 *  itself.
 *
 *  Doing things this way lets us swap out the underlying handler – which may
 *  happen regularly - without needing to rebind the event listener.
 *
 */ const handleEvent = (event)=>{
    const { currentTarget, type } = event;
    const handler = currentTarget[meta].handlers.get(type);
    handler(event);
};
// ATTRIBUTE SPECIAL CASES -----------------------------------------------------
/* @__NO_SIDE_EFFECTS__ */ const syncedBooleanAttribute = (name)=>{
    return {
        added (node) {
            node[name] = true;
        },
        removed (node) {
            node[name] = false;
        }
    };
};
/* @__NO_SIDE_EFFECTS__ */ const syncedAttribute = (name)=>{
    return {
        added (node, value) {
            node[name] = value;
        }
    };
};
const SYNCED_ATTRIBUTES = {
    checked: syncedBooleanAttribute("checked"),
    selected: syncedBooleanAttribute("selected"),
    value: syncedAttribute("value"),
    autofocus: {
        added (node) {
            queueMicrotask(()=>{
                node.focus?.();
            });
        }
    },
    autoplay: {
        added (node) {
            try {
                node.play?.();
            } catch (e) {
                console.error(e);
            }
        }
    }
};

},{"../../../houdini/houdini.mjs":"e94ou","./vnode.mjs":"j2vnp","./vattr.mjs":"jrrcC","./patch.mjs":"31vMv","./path.mjs":"351yX","../internals/list.ffi.mjs":"hGVW1","../internals/constants.ffi.mjs":"8U0vL","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"hGVW1":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "toList", ()=>toList);
parcelHelpers.export(exports, "iterate", ()=>iterate);
parcelHelpers.export(exports, "append", ()=>append);
var _gleamMjs = require("../../gleam.mjs");
var _constantsMjs = require("./constants.mjs");
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
const toList = (arr)=>arr.reduceRight((xs, x)=>(0, _gleamMjs.List$NonEmpty)(x, xs), (0, _constantsMjs.empty_list));
const iterate = (list, callback)=>{
    if (Array.isArray(list)) for(let i = 0; i < list.length; i++)callback(list[i]);
    else if (list) for(list; (0, _gleamMjs.List$NonEmpty$rest)(list); list = (0, _gleamMjs.List$NonEmpty$rest)(list))callback((0, _gleamMjs.List$NonEmpty$first)(list));
};
const append = (a, b)=>{
    if (!(0, _gleamMjs.List$NonEmpty$rest)(a)) return b;
    else if (!(0, _gleamMjs.List$NonEmpty$rest)(b)) return a;
    else return (0, _listMjs.append)(a, b);
};

},{"../../gleam.mjs":"jNPQG","./constants.mjs":"gKFR6","../../../gleam_stdlib/gleam/list.mjs":"8dUwY","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"8U0vL":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "NAMESPACE_HTML", ()=>NAMESPACE_HTML);
parcelHelpers.export(exports, "ELEMENT_NODE", ()=>ELEMENT_NODE);
parcelHelpers.export(exports, "TEXT_NODE", ()=>TEXT_NODE);
parcelHelpers.export(exports, "COMMENT_NODE", ()=>COMMENT_NODE);
parcelHelpers.export(exports, "DOCUMENT_FRAGMENT_NODE", ()=>DOCUMENT_FRAGMENT_NODE);
parcelHelpers.export(exports, "SUPPORTS_MOVE_BEFORE", ()=>SUPPORTS_MOVE_BEFORE);
const NAMESPACE_HTML = "http://www.w3.org/1999/xhtml";
const ELEMENT_NODE = 1;
const TEXT_NODE = 3;
const COMMENT_NODE = 8;
const DOCUMENT_FRAGMENT_NODE = 11;
const SUPPORTS_MOVE_BEFORE = /* @__PURE__ */ !!globalThis.HTMLElement?.prototype?.moveBefore;

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"k2cHU":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "virtualise", ()=>virtualise);
var _gleamMjs = require("../../gleam.mjs");
var _elementMjs = require("../element.mjs");
var _keyedMjs = require("../element/keyed.mjs");
var _attributeMjs = require("../attribute.mjs");
var _reconcilerFfiMjs = require("./reconciler.ffi.mjs");
var _vnodeMjs = require("./vnode.mjs");
var _constantsMjs = require("../internals/constants.mjs");
var _constantsFfiMjs = require("../internals/constants.ffi.mjs");
const virtualise = (root)=>{
    // no matter what, we want to initialise the metadata for our root element.
    // we pass an empty stringh here as the index to make sure that the root node
    // does not have a path.
    const rootMeta = (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.element_kind), null, root, 0, null);
    for(let child = root.firstChild; child; child = child.nextSibling){
        const result = virtualiseChild(rootMeta, root, child, 0);
        // lustre view functions always return a single root element inside the root.
        // even if we could virtualise multiple children, we will ignore them and
        // return the first child as the one we'll take over.
        //
        // A top-level key is impossible and always ignored.
        if (result) return result.vnode;
    }
    // no virtualisable children, we can empty the node and return our default text node.
    const placeholder = globalThis.document.createTextNode("");
    (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.text_kind), rootMeta, placeholder, 0, null);
    root.insertBefore(placeholder, root.firstChild);
    return (0, _elementMjs.none)();
};
const virtualiseChild = (meta, domParent, child, index)=>{
    if (child.nodeType === (0, _constantsFfiMjs.COMMENT_NODE)) {
        const data = child.data.trim();
        if (data.startsWith("lustre:fragment")) return virtualiseFragment(meta, domParent, child, index);
        if (data.startsWith("lustre:map")) return virtualiseMap(meta, domParent, child, index);
        if (data.startsWith("lustre:memo")) return virtualiseMemo(meta, domParent, child, index);
        return null;
    }
    if (child.nodeType === (0, _constantsFfiMjs.ELEMENT_NODE)) return virtualiseElement(meta, child, index);
    if (child.nodeType === (0, _constantsFfiMjs.TEXT_NODE)) return virtualiseText(meta, child, index);
    return null;
};
const virtualiseElement = (metaParent, node, index)=>{
    const key = node.getAttribute("data-lustre-key") ?? "";
    if (key) node.removeAttribute("data-lustre-key");
    const meta = (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.element_kind), metaParent, node, index, key);
    const tag = node.localName;
    const namespace = node.namespaceURI;
    const isHtmlElement = !namespace || namespace === (0, _constantsFfiMjs.NAMESPACE_HTML);
    if (isHtmlElement && INPUT_ELEMENTS.includes(tag)) virtualiseInputEvents(tag, node);
    const attributes = virtualiseAttributes(node);
    const children = [];
    for(let childNode = node.firstChild; childNode;){
        const child = virtualiseChild(meta, node, childNode, children.length);
        if (child) {
            children.push([
                child.key,
                child.vnode
            ]);
            childNode = child.next;
        } else childNode = childNode.nextSibling;
    }
    const vnode = isHtmlElement ? (0, _keyedMjs.element)(tag, attributes, toList(children)) : (0, _keyedMjs.namespaced)(namespace, tag, attributes, toList(children));
    return childResult(key, vnode, node.nextSibling);
};
const virtualiseText = (meta, node, index)=>{
    (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.text_kind), meta, node, index, null);
    return childResult("", (0, _elementMjs.text)(node.data), node.nextSibling);
};
const virtualiseFragment = (metaParent, domParent, node, index)=>{
    const key = parseKey(node.data);
    const meta = (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.fragment_kind), metaParent, node, index, key);
    const children = [];
    node = node.nextSibling;
    while(node && (node.nodeType !== (0, _constantsFfiMjs.COMMENT_NODE) || node.data.trim() !== "/lustre:fragment")){
        const child = virtualiseChild(meta, domParent, node, children.length);
        if (child) {
            children.push([
                child.key,
                child.vnode
            ]);
            node = child.next;
        } else node = node.nextSibling;
    }
    meta.endNode = node;
    const vnode = (0, _keyedMjs.fragment)(toList(children));
    return childResult(key, vnode, node?.nextSibling);
};
const virtualiseMap = (metaParent, domParent, node, index)=>{
    const key = parseKey(node.data);
    const meta = (0, _reconcilerFfiMjs.insertMetadataChild)((0, _vnodeMjs.map_kind), metaParent, node, index, key);
    const child = virtualiseNextChild(meta, domParent, node, 0);
    if (!child) return null;
    const vnode = (0, _elementMjs.map)(child.vnode, (x)=>x);
    return childResult(key, vnode, child.next);
};
const virtualiseMemo = (meta, domParent, node, index)=>{
    const key = parseKey(node.data);
    // Memo nodes are transparent - they don't create metadata nodes!
    // Just virtualise the child directly with the parent metadata
    const child = virtualiseNextChild(meta, domParent, node, index);
    if (!child) return null;
    domParent.removeChild(node);
    // We cannot recover the dependencies -
    // so we add an anonymous object here that for sure compares falsy with
    // anything the user will pass us.
    const vnode = (0, _elementMjs.memo)(toList([
        (0, _elementMjs.ref)({})
    ]), ()=>child.vnode);
    return childResult(key, vnode, child.next);
};
const virtualiseNextChild = (meta, domParent, node, index)=>{
    while(true){
        node = node.nextSibling;
        if (!node) return null;
        const child = virtualiseChild(meta, domParent, node, index);
        if (child) return child;
    }
};
const childResult = (key, vnode, next)=>{
    return {
        key,
        vnode,
        next
    };
};
const virtualiseAttributes = (node)=>{
    const attributes = [];
    for(let i = 0; i < node.attributes.length; i++){
        const attr = node.attributes[i];
        if (attr.name !== "xmlns") attributes.push((0, _attributeMjs.attribute)(attr.localName, attr.value));
    }
    return toList(attributes);
};
const INPUT_ELEMENTS = [
    "input",
    "select",
    "textarea"
];
const virtualiseInputEvents = (tag, node)=>{
    const value = node.value;
    const checked = node.checked;
    // For inputs that reflect their default state (eg not checked for checkboxes
    // and radios, empty for all other inputs) then we don't need to schedule any
    // virtual events.
    if (tag === "input" && node.type === "checkbox" && !checked) return;
    if (tag === "input" && node.type === "radio" && !checked) return;
    if (node.type !== "checkbox" && node.type !== "radio" && !value) return;
    // We schedule a microtask instead of dispatching the events immediately to
    // give the runtime a chance to finish virtualising the DOM and set up the
    // runtime.
    //
    // Microtasks are flushed once the current task has completed, and will block
    // the browser from painting until the queue is empty, so we can be sure that
    // these events will be processed before the user sees the first render.
    queueMicrotask(()=>{
        // Since the first patch will have overridden our values, we will reset them
        // here and trigger events, which the runtime can then pick up.
        node.value = value;
        node.checked = checked;
        node.dispatchEvent(new Event("input", {
            bubbles: true
        }));
        node.dispatchEvent(new Event("change", {
            bubbles: true
        }));
        // User apps may be using semi-controlled inputs where they listen to blur
        // events to save the value rather than using the input event. To account for
        // those, we dispatch a blur event if the input is not currently focused.
        if (globalThis.document.activeElement !== node) node.dispatchEvent(new Event("blur", {
            bubbles: true
        }));
    });
};
const parseKey = (data)=>{
    const keyMatch = data.match(/key="([^"]*)"/);
    if (!keyMatch) return "";
    return unescapeKey(keyMatch[1]);
};
const unescapeKey = (key)=>{
    return key.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&#39;/g, "'");
};
const toList = (arr)=>arr.reduceRight((xs, x)=>(0, _gleamMjs.List$NonEmpty)(x, xs), (0, _constantsMjs.empty_list));

},{"../../gleam.mjs":"jNPQG","../element.mjs":"2XxJ4","../element/keyed.mjs":"cxGER","../attribute.mjs":"faRXj","./reconciler.ffi.mjs":"5QzuP","./vnode.mjs":"j2vnp","../internals/constants.mjs":"gKFR6","../internals/constants.ffi.mjs":"8U0vL","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"cxGER":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Render a _keyed_ element with the given tag. Each child is assigned a unique
 * key, which Lustre uses to identify the element in the DOM. This is useful when
 * a single child can be moved around such as in a to-do list, or when elements
 * are frequently added or removed.
 *
 * > **Note**: the key for each child must be unique within the list of children,
 * > but it doesn't have to be unique across the whole application. It's fine to
 * > use the same key in different lists.
 */ parcelHelpers.export(exports, "element", ()=>element);
/**
 * Render a _keyed_ element with the given namespace and tag. Each child is
 * assigned a unique key, which Lustre uses to identify the element in the DOM.
 * This is useful when a single child can be moved around such as in a to-do
 * list, or when elements are frequently added or removed.
 *
 * > **Note**: the key for each child must be unique within the list of children,
 * > but it doesn't have to be unique across the whole application. It's fine to
 * > use the same key in different lists.
 */ parcelHelpers.export(exports, "namespaced", ()=>namespaced);
/**
 * Render a _keyed_ fragment. Each child is assigned a unique key, which Lustre
 * uses to identify the element in the DOM. This is useful when a single child
 * can be moved around such as in a to-do list, or when elements are frequently
 * added or removed.
 *
 * > **Note**: the key for each child must be unique within the list of children,
 * > but it doesn't have to be unique across the whole application. It's fine to
 * > use the same key in different lists.
 */ parcelHelpers.export(exports, "fragment", ()=>fragment);
parcelHelpers.export(exports, "ul", ()=>ul);
parcelHelpers.export(exports, "ol", ()=>ol);
parcelHelpers.export(exports, "div", ()=>div);
parcelHelpers.export(exports, "tbody", ()=>tbody);
parcelHelpers.export(exports, "dl", ()=>dl);
var _listMjs = require("../../../gleam_stdlib/gleam/list.mjs");
var _gleamMjs = require("../../gleam.mjs");
var _attributeMjs = require("../../lustre/attribute.mjs");
var _elementMjs = require("../../lustre/element.mjs");
var _constantsMjs = require("../../lustre/internals/constants.mjs");
var _mutableMapMjs = require("../../lustre/internals/mutable_map.mjs");
var _vnodeMjs = require("../../lustre/vdom/vnode.mjs");
function do_extract_keyed_children(loop$key_children_pairs, loop$keyed_children, loop$children) {
    while(true){
        let key_children_pairs = loop$key_children_pairs;
        let keyed_children = loop$keyed_children;
        let children = loop$children;
        if (key_children_pairs instanceof (0, _gleamMjs.Empty)) return [
            keyed_children,
            _listMjs.reverse(children)
        ];
        else {
            let rest = key_children_pairs.tail;
            let key = key_children_pairs.head[0];
            let element$1 = key_children_pairs.head[1];
            let keyed_element = _vnodeMjs.to_keyed(key, element$1);
            let _block;
            if (key === "") _block = keyed_children;
            else _block = _mutableMapMjs.insert(keyed_children, key, keyed_element);
            let keyed_children$1 = _block;
            let children$1 = (0, _gleamMjs.prepend)(keyed_element, children);
            loop$key_children_pairs = rest;
            loop$keyed_children = keyed_children$1;
            loop$children = children$1;
        }
    }
}
function extract_keyed_children(children) {
    return do_extract_keyed_children(children, _mutableMapMjs.new$(), _constantsMjs.empty_list);
}
function element(tag, attributes, children) {
    let $ = extract_keyed_children(children);
    let keyed_children;
    let children$1;
    keyed_children = $[0];
    children$1 = $[1];
    return _vnodeMjs.element("", "", tag, attributes, children$1, keyed_children, false, _vnodeMjs.is_void_html_element(tag, ""));
}
function namespaced(namespace, tag, attributes, children) {
    let $ = extract_keyed_children(children);
    let keyed_children;
    let children$1;
    keyed_children = $[0];
    children$1 = $[1];
    return _vnodeMjs.element("", namespace, tag, attributes, children$1, keyed_children, false, _vnodeMjs.is_void_html_element(tag, namespace));
}
function fragment(children) {
    let $ = extract_keyed_children(children);
    let keyed_children;
    let children$1;
    keyed_children = $[0];
    children$1 = $[1];
    return _vnodeMjs.fragment("", children$1, keyed_children);
}
function ul(attributes, children) {
    return element("ul", attributes, children);
}
function ol(attributes, children) {
    return element("ol", attributes, children);
}
function div(attributes, children) {
    return element("div", attributes, children);
}
function tbody(attributes, children) {
    return element("tbody", attributes, children);
}
function dl(attributes, children) {
    return element("dl", attributes, children);
}

},{"../../../gleam_stdlib/gleam/list.mjs":"8dUwY","../../gleam.mjs":"jNPQG","../../lustre/attribute.mjs":"faRXj","../../lustre/element.mjs":"2XxJ4","../../lustre/internals/constants.mjs":"gKFR6","../../lustre/internals/mutable_map.mjs":"6NvMa","../../lustre/vdom/vnode.mjs":"j2vnp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"cVaD8":[function(require,module,exports,__globalThis) {
// IMPORTS ---------------------------------------------------------------------
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
//
parcelHelpers.export(exports, "Spa", ()=>Spa);
parcelHelpers.export(exports, "start", ()=>start);
var _gleamMjs = require("../../../gleam.mjs");
var _lustreMjs = require("../../../lustre.mjs");
var _runtimeFfiMjs = require("./runtime.ffi.mjs");
var _runtimeMjs = require("../server/runtime.mjs");
class Spa {
    #runtime;
    constructor(root, [init, effects], update, view){
        this.#runtime = new (0, _runtimeFfiMjs.Runtime)(root, [
            init,
            effects
        ], view, update);
    }
    send(message) {
        if ((0, _runtimeMjs.Message$isEffectDispatchedMessage)(message)) this.dispatch(message.message, false);
        else if ((0, _runtimeMjs.Message$isEffectEmitEvent)(message)) this.emit(message.name, message.data);
        else (0, _runtimeMjs.Message$isSystemRequestedShutdown)(message);
    }
    dispatch(msg) {
        this.#runtime.dispatch(msg);
    }
    emit(event, data) {
        this.#runtime.emit(event, data);
    }
}
const start = ({ init, update, view }, selector, flags)=>{
    if (!(0, _runtimeFfiMjs.is_browser)()) return (0, _gleamMjs.Result$Error)((0, _lustreMjs.Error$NotABrowser)());
    const root = selector instanceof HTMLElement ? selector : globalThis.document.querySelector(selector);
    if (!root) return (0, _gleamMjs.Result$Error)((0, _lustreMjs.Error$ElementNotFound)(selector));
    return (0, _gleamMjs.Result$Ok)(new Spa(root, init(flags), update, view));
};

},{"../../../gleam.mjs":"jNPQG","../../../lustre.mjs":"9FST8","./runtime.ffi.mjs":"eto4y","../server/runtime.mjs":"8rUwG","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"kJVZ5":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
//
parcelHelpers.export(exports, "Runtime", ()=>Runtime);
parcelHelpers.export(exports, "start", ()=>start);
parcelHelpers.export(exports, "send", ()=>send);
var _gleamMjs = require("../../../gleam.mjs");
var _decodeMjs = require("../../../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _dictMjs = require("../../../../gleam_stdlib/gleam/dict.mjs");
var _optionMjs = require("../../../../gleam_stdlib/gleam/option.mjs");
var _diffMjs = require("../../vdom/diff.mjs");
var _cacheMjs = require("../../vdom/cache.mjs");
var _equalsFfiMjs = require("../../internals/equals.ffi.mjs");
var _runtimeMjs = require("./runtime.mjs");
var _componentMjs = require("../../component.mjs");
var _effectMjs = require("../../effect.mjs");
var _transportMjs = require("../transport.mjs");
var _listFfiMjs = require("../../internals/list.ffi.mjs");
class Runtime {
    #model;
    #update;
    #view;
    #config;
    #vdom;
    #cache;
    #providers = _dictMjs.new$();
    #callbacks = /* @__PURE__ */ new Set();
    constructor(_, init, update, view, config1, start_arguments){
        const [model, effects] = init(start_arguments);
        this.#model = model;
        this.#update = update;
        this.#view = view;
        this.#config = config1;
        this.#vdom = this.#view(this.#model);
        this.#cache = _cacheMjs.from_node(this.#vdom);
        this.#handle_effect(effects);
    }
    send(msg) {
        if ((0, _runtimeMjs.Message$isClientDispatchedMessage)(msg)) {
            const { message } = msg;
            const next = this.#handle_client_message(message);
            const diff = _diffMjs.diff(this.#cache, this.#vdom, next);
            this.#vdom = next;
            this.#cache = diff.cache;
            this.broadcast(_transportMjs.reconcile(diff.patch, _cacheMjs.memos(diff.cache)));
        } else if ((0, _runtimeMjs.Message$isClientRegisteredCallback)(msg)) {
            const { callback } = msg;
            this.#callbacks.add(callback);
            callback(_transportMjs.mount(this.#config.open_shadow_root, this.#config.adopt_styles, _dictMjs.keys(this.#config.attributes), _dictMjs.keys(this.#config.properties), _dictMjs.keys(this.#config.contexts), this.#providers, this.#vdom, _cacheMjs.memos(this.#cache)));
            if (_optionMjs.Option$isSome(config.on_connect)) this.#dispatch(_optionMjs.Option$Some$0(config.on_connect));
        } else if ((0, _runtimeMjs.Message$isClientDeregisteredCallback)(msg)) {
            const { callback } = msg;
            this.#callbacks.delete(callback);
            if (_optionMjs.Option$isSome(config.on_disconnect)) this.#dispatch(_optionMjs.Option$Some$0(config.on_disconnect));
        } else if ((0, _runtimeMjs.Message$isEffectDispatchedMessage)(msg)) {
            const { message } = msg;
            const [model, effect] = this.#update(this.#model, message);
            const next = this.#view(model);
            const diff = _diffMjs.diff(this.#cache, this.#vdom, next);
            this.#handle_effect(effect);
            this.#model = model;
            this.#vdom = next;
            this.#cache = diff.cache;
            this.broadcast(_transportMjs.reconcile(diff.patch, _cacheMjs.memos(diff.cache)));
        } else if ((0, _runtimeMjs.Message$isEffectEmitEvent)(msg)) {
            const { name, data } = msg;
            this.broadcast(_transportMjs.emit(name, data));
        } else if ((0, _runtimeMjs.Message$isEffectProvidedValue)(msg)) {
            const { key, value } = msg;
            const existing = _dictMjs.get(this.#providers, key);
            // we do not need to broadcast an update if the provided value is the same.
            if ((0, _gleamMjs.Result$isOk)(existing) && (0, _equalsFfiMjs.isEqual)((0, _gleamMjs.Result$Ok$0)(existing), value)) return;
            this.#providers = _dictMjs.insert(this.#providers, key, value);
            this.broadcast(_transportMjs.provide(key, value));
        } else if ((0, _runtimeMjs.Message$isSystemRequestedShutdown)(msg)) {
            this.#model = null;
            this.#update = null;
            this.#view = null;
            this.#config = null;
            this.#vdom = null;
            this.#cache = null;
            this.#providers = null;
            this.#callbacks.clear();
        }
    }
    broadcast(msg) {
        for (const callback of this.#callbacks)callback(msg);
    }
    #handle_client_message(msg) {
        if ((0, _transportMjs.ServerMessage$isBatch)(msg)) {
            const { messages } = msg;
            let model = this.#model;
            let effect = _effectMjs.none();
            for(let list = messages; (0, _gleamMjs.List$NonEmpty$rest)(list); list = (0, _gleamMjs.List$NonEmpty$rest)(list)){
                const result = this.#handle_client_message((0, _gleamMjs.List$NonEmpty$first)(list));
                if ((0, _gleamMjs.Result$isOk)(result)) {
                    model = (0, _gleamMjs.Result$Ok$0)(result)[0];
                    effect = _effectMjs.batch((0, _listFfiMjs.toList)([
                        effect,
                        (0, _gleamMjs.Result$Ok$0)(result)[1]
                    ]));
                    break;
                }
            }
            this.#handle_effect(effect);
            this.#model = model;
            return this.#view(model);
        } else if ((0, _transportMjs.ServerMessage$isAttributeChanged)(msg)) {
            const { name, value } = msg;
            const result = this.#handle_attribute_change(name, value);
            if (!(0, _gleamMjs.Result$isOk)(result)) return this.#vdom;
            return this.#dispatch((0, _gleamMjs.Result$Ok$0)(result));
        } else if ((0, _transportMjs.ServerMessage$isPropertyChanged)(msg)) {
            const { name, value } = msg;
            const result = this.#handle_properties_change(name, value);
            if (!(0, _gleamMjs.Result$isOk)(result)) return this.#vdom;
            return this.#dispatch((0, _gleamMjs.Result$Ok$0)(result));
        } else if ((0, _transportMjs.ServerMessage$isEventFired)(msg)) {
            const { path, name, event } = msg;
            const [cache, result] = _cacheMjs.handle(this.#cache, path, name, event);
            this.#cache = cache;
            if (!(0, _gleamMjs.Result$isOk)(result)) return this.#vdom;
            const { message } = (0, _gleamMjs.Result$Ok$0)(result);
            return this.#dispatch(message);
        } else if ((0, _transportMjs.ServerMessage$isContextProvided)(msg)) {
            const { key, value } = msg;
            let result = _dictMjs.get(this.#config.contexts, key);
            if (!(0, _gleamMjs.Result$isOk)(result)) return this.#vdom;
            result = _decodeMjs.run(value, (0, _gleamMjs.Result$Ok$0)(result));
            if (!(0, _gleamMjs.Result$isOk)(result)) return this.#vdom;
            return this.#dispatch((0, _gleamMjs.Result$Ok$0)(result));
        }
    }
    #dispatch(msg) {
        const [model, effects] = this.#update(this.#model, msg);
        this.#handle_effect(effects);
        this.#model = model;
        return this.#view(this.#model);
    }
    #handle_attribute_change(name, value) {
        const result = _dictMjs.get(this.#config.attributes, name);
        if (!(0, _gleamMjs.Result$isOk)(result)) return result;
        return (0, _gleamMjs.Result$Ok$0)(result)(value);
    }
    #handle_properties_change(name, value) {
        const result = _dictMjs.get(this.#config.properties, name);
        if (!(0, _gleamMjs.Result$isOk)(result)) return result;
        return (0, _gleamMjs.Result$Ok$0)(result)(value);
    }
    #handle_effect(effect) {
        const dispatch = (message)=>this.send((0, _runtimeMjs.Message$EffectDispatchedMessage)(message));
        const emit = (name, data)=>this.send((0, _runtimeMjs.Message$EffectEmitEvent)(name, data));
        const select = ()=>undefined;
        const internals = ()=>undefined;
        const provide = (key, value)=>this.send((0, _runtimeMjs.Message$EffectProvidedValue)(key, value));
        globalThis.queueMicrotask(()=>{
            _effectMjs.perform(effect, dispatch, emit, select, internals, provide);
        });
    }
}
const start = (app, start_arguments)=>{
    const config1 = _componentMjs.to_server_component_config(app.config);
    return (0, _gleamMjs.Result$Ok)(new Runtime(app.init, app.update, app.view, config1, start_arguments));
};
const send = (runtime, message)=>{
    runtime.send(message);
};

},{"../../../gleam.mjs":"jNPQG","../../../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../../../gleam_stdlib/gleam/dict.mjs":"b8yrU","../../../../gleam_stdlib/gleam/option.mjs":"aWtoH","../../vdom/diff.mjs":"iOcdA","../../vdom/cache.mjs":"aEh50","../../internals/equals.ffi.mjs":"2LTPm","./runtime.mjs":"8rUwG","../../component.mjs":"k3Cmy","../../effect.mjs":"iAEPi","../transport.mjs":"9jG6q","../../internals/list.ffi.mjs":"hGVW1","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"29g6I":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
/**
 * Dispatches a custom message from a Lustre component. This lets components
 * communicate with their parents the same way native DOM elements do.
 *
 * Any JSON-serialisable payload can be attached as additional data for any
 * event listeners to decode. This data will be on the event's `detail` property.
 */ parcelHelpers.export(exports, "emit", ()=>emit);
/**
 * Listens for the given event and then runs the given decoder on the event
 * object. If the decoder succeeds, the decoded event is dispatched to your
 * application's `update` function. If it fails, the event is silently ignored.
 *
 * The event name is typically an all-lowercase string such as "click" or "mousemove".
 * If you're listening for non-standard events (like those emitted by a custom
 * element) their event names might be slightly different.
 *
 * > **Note**: if you are developing a server component, it is important to also
 * > use [`server_component.include`](./server_component.html#include) to state
 * > which properties of the event you need to be sent to the server.
 */ parcelHelpers.export(exports, "on", ()=>on);
/**
 * Listens for the given event and then runs the given decoder on the event
 * object. This decoder is capable of _conditionally_ stopping propagation or
 * preventing the default behaviour of the event by returning a `Handler` record
 * with the appropriate flags set. This makes it possible to write event handlers
 * for more-advanced scenarios such as handling specific key presses.
 *
 * > **Note**: it is not possible to conditionally stop propagation or prevent
 * > the default behaviour of an event when using _server components_. Your event
 * > handler runs on the server, far away from the browser!
 *
 * > **Note**: if you are developing a server component, it is important to also
 * > use [`server_component.include`](./server_component.html#include) to state
 * > which properties of the event you need to be sent to the server.
 */ parcelHelpers.export(exports, "advanced", ()=>advanced);
/**
 * Construct a [`Handler`](#Handler) that can be used with [`advanced`](#advanced)
 * to conditionally stop propagation or prevent the default behaviour of an event.
 */ parcelHelpers.export(exports, "handler", ()=>handler);
/**
 * Indicate that the event should have its default behaviour cancelled. This is
 * equivalent to calling `event.preventDefault()` in JavaScript.
 *
 * > **Note**: this will override the conditional behaviour of an event handler
 * > created with [`advanced`](#advanced).
 */ parcelHelpers.export(exports, "prevent_default", ()=>prevent_default);
/**
 * Indicate that the event should not propagate to parent elements. This is
 * equivalent to calling `event.stopPropagation()` in JavaScript.
 *
 * > **Note**: this will override the conditional behaviour of an event handler
 * > created with [`advanced`](#advanced).
 */ parcelHelpers.export(exports, "stop_propagation", ()=>stop_propagation);
/**
 * Use Lustre's built-in event debouncing to wait a delay after a burst of
 * events before dispatching the most recent one. You can visualise debounced
 * events like so:
 *
 * ```
 *  original : --a-b-cd--e----------f--------
 * debounced : ---------------e----------f---
 * ```
 *
 * This is particularly useful for server components where many events in quick
 * succession can introduce problems because of network latency.
 *
 * The unit of `delay` is millisecond, same as JavaScript's `setTimeout`.
 *
 * ### Example:
 *
 * ```gleam
 * type Msg {
 *     UserInputText(String)
 * }
 *
 * html.input([event.debounce(event.on_input(fn(v) { UserInputText(v) }), 200)])
 * ```
 *
 * > **Note**: debounced events inherently introduce latency. Try to consider
 * > typical interaction patterns and experiment with different delays to balance
 * > responsiveness and update frequency.
 */ parcelHelpers.export(exports, "debounce", ()=>debounce);
/**
 * Use Lustre's built-in event throttling to restrict the number of events
 * that can be dispatched in a given time period. You can visualise throttled
 * events like so:
 *
 * ```
 * original : --a-b-cd--e----------f--------
 * throttled : -a------ e----------e--------
 * ```
 *
 * This is particularly useful for server components where many events in quick
 * succession can introduce problems because of network latency.
 *
 * The unit of `delay` is millisecond, same as JavaScript's `setTimeout`.
 *
 * > **Note**: throttled events inherently reduce precision. Try to consider
 * > typical interaction patterns and experiment with different delays to balance
 * > responsiveness and update frequency.
 */ parcelHelpers.export(exports, "throttle", ()=>throttle);
/**
 *
 */ parcelHelpers.export(exports, "on_click", ()=>on_click);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_down", ()=>on_mouse_down);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_up", ()=>on_mouse_up);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_enter", ()=>on_mouse_enter);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_leave", ()=>on_mouse_leave);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_over", ()=>on_mouse_over);
/**
 *
 */ parcelHelpers.export(exports, "on_mouse_out", ()=>on_mouse_out);
/**
 * Listens for key presses on an element, and dispatches a message with the
 * current key being pressed.
 */ parcelHelpers.export(exports, "on_keypress", ()=>on_keypress);
/**
 * Listens for key down events on an element, and dispatches a message with the
 * current key being pressed.
 */ parcelHelpers.export(exports, "on_keydown", ()=>on_keydown);
/**
 * Listens for key up events on an element, and dispatches a message with the
 * current key being released.
 */ parcelHelpers.export(exports, "on_keyup", ()=>on_keyup);
/**
 * Listens for input events on elements such as `<input>`, `<textarea>` and
 * `<select>`. This handler automatically decodes the string value of the input
 * and passes it to the given message function. This is commonly used to
 * implement [controlled inputs](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 */ parcelHelpers.export(exports, "on_input", ()=>on_input);
/**
 * Listens for change events on elements such as `<input>`, `<textarea>` and
 * `<select>`. This handler automatically decodes the string value of the input
 * and passes it to the given message function. This is commonly used to
 * implement [controlled inputs](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 */ parcelHelpers.export(exports, "on_change", ()=>on_change);
/**
 * Listens for change events on `<input type="checkbox">` elements. This handler
 * automatically decodes the boolean value of the checkbox and passes it to
 * the given message function. This is commonly used to implement
 * [controlled inputs](https://github.com/lustre-labs/lustre/blob/main/pages/hints/controlled-vs-uncontrolled-inputs.md).
 */ parcelHelpers.export(exports, "on_check", ()=>on_check);
/**
 * Listens for submit events on a `<form>` element and receives a list of
 * name/value pairs for each field in the form. Files are not included in this
 * list: if you need them, you can write your own handler for the `"submit"`
 * event and decode the non-standard `detail.formData` property manually.
 *
 * This handler is best paired with the [`formal`](https://hexdocs.pm/formal/)
 * package which lets you process form submissions in a type-safe way.
 *
 * This will automatically call [`prevent_default`](#prevent_default) to stop
 * the browser's native form submission. In a Lustre app you'll want to handle
 * that yourself as an [`Effect`](./effect.html#Effect).
 */ parcelHelpers.export(exports, "on_submit", ()=>on_submit);
parcelHelpers.export(exports, "on_focus", ()=>on_focus);
parcelHelpers.export(exports, "on_blur", ()=>on_blur);
var _jsonMjs = require("../../gleam_json/gleam/json.mjs");
var _decodeMjs = require("../../gleam_stdlib/gleam/dynamic/decode.mjs");
var _intMjs = require("../../gleam_stdlib/gleam/int.mjs");
var _pairMjs = require("../../gleam_stdlib/gleam/pair.mjs");
var _resultMjs = require("../../gleam_stdlib/gleam/result.mjs");
var _gleamMjs = require("../gleam.mjs");
var _attributeMjs = require("../lustre/attribute.mjs");
var _effectMjs = require("../lustre/effect.mjs");
var _constantsMjs = require("../lustre/internals/constants.mjs");
var _vattrMjs = require("../lustre/vdom/vattr.mjs");
function emit(event, data) {
    return _effectMjs.event(event, data);
}
function on(name, handler) {
    return _vattrMjs.event(name, _decodeMjs.map(handler, (msg)=>{
        return new (0, _vattrMjs.Handler)(false, false, msg);
    }), _constantsMjs.empty_list, _vattrMjs.never, _vattrMjs.never, 0, 0);
}
function advanced(name, handler) {
    return _vattrMjs.event(name, handler, _constantsMjs.empty_list, _vattrMjs.possible, _vattrMjs.possible, 0, 0);
}
function handler(message, prevent_default, stop_propagation) {
    return new (0, _vattrMjs.Handler)(prevent_default, stop_propagation, message);
}
function prevent_default(event) {
    if (event instanceof (0, _vattrMjs.Event)) return new (0, _vattrMjs.Event)(event.kind, event.name, event.handler, event.include, _vattrMjs.always, event.stop_propagation, event.debounce, event.throttle);
    else return event;
}
function stop_propagation(event) {
    if (event instanceof (0, _vattrMjs.Event)) return new (0, _vattrMjs.Event)(event.kind, event.name, event.handler, event.include, event.prevent_default, _vattrMjs.always, event.debounce, event.throttle);
    else return event;
}
function debounce(event, delay) {
    if (event instanceof (0, _vattrMjs.Event)) return new (0, _vattrMjs.Event)(event.kind, event.name, event.handler, event.include, event.prevent_default, event.stop_propagation, _intMjs.max(0, delay), event.throttle);
    else return event;
}
function throttle(event, delay) {
    if (event instanceof (0, _vattrMjs.Event)) return new (0, _vattrMjs.Event)(event.kind, event.name, event.handler, event.include, event.prevent_default, event.stop_propagation, event.debounce, _intMjs.max(0, delay));
    else return event;
}
function on_click(msg) {
    return on("click", _decodeMjs.success(msg));
}
function on_mouse_down(msg) {
    return on("mousedown", _decodeMjs.success(msg));
}
function on_mouse_up(msg) {
    return on("mouseup", _decodeMjs.success(msg));
}
function on_mouse_enter(msg) {
    return on("mouseenter", _decodeMjs.success(msg));
}
function on_mouse_leave(msg) {
    return on("mouseleave", _decodeMjs.success(msg));
}
function on_mouse_over(msg) {
    return on("mouseover", _decodeMjs.success(msg));
}
function on_mouse_out(msg) {
    return on("mouseout", _decodeMjs.success(msg));
}
function on_keypress(msg) {
    return on("keypress", _decodeMjs.field("key", _decodeMjs.string, (key)=>{
        let _pipe = key;
        let _pipe$1 = msg(_pipe);
        return _decodeMjs.success(_pipe$1);
    }));
}
function on_keydown(msg) {
    return on("keydown", _decodeMjs.field("key", _decodeMjs.string, (key)=>{
        let _pipe = key;
        let _pipe$1 = msg(_pipe);
        return _decodeMjs.success(_pipe$1);
    }));
}
function on_keyup(msg) {
    return on("keyup", _decodeMjs.field("key", _decodeMjs.string, (key)=>{
        let _pipe = key;
        let _pipe$1 = msg(_pipe);
        return _decodeMjs.success(_pipe$1);
    }));
}
function on_input(msg) {
    return on("input", _decodeMjs.subfield((0, _gleamMjs.toList)([
        "target",
        "value"
    ]), _decodeMjs.string, (value)=>{
        return _decodeMjs.success(msg(value));
    }));
}
function on_change(msg) {
    return on("change", _decodeMjs.subfield((0, _gleamMjs.toList)([
        "target",
        "value"
    ]), _decodeMjs.string, (value)=>{
        return _decodeMjs.success(msg(value));
    }));
}
function on_check(msg) {
    return on("change", _decodeMjs.subfield((0, _gleamMjs.toList)([
        "target",
        "checked"
    ]), _decodeMjs.bool, (checked)=>{
        return _decodeMjs.success(msg(checked));
    }));
}
function formdata_decoder() {
    let string_value_decoder = _decodeMjs.field(0, _decodeMjs.string, (key)=>{
        return _decodeMjs.field(1, _decodeMjs.one_of(_decodeMjs.map(_decodeMjs.string, (var0)=>{
            return new (0, _gleamMjs.Ok)(var0);
        }), (0, _gleamMjs.toList)([
            _decodeMjs.success(_constantsMjs.error_nil)
        ])), (value)=>{
            let _pipe = value;
            let _pipe$1 = _resultMjs.map(_pipe, (_capture)=>{
                return _pairMjs.new$(key, _capture);
            });
            return _decodeMjs.success(_pipe$1);
        });
    });
    let _pipe = string_value_decoder;
    let _pipe$1 = _decodeMjs.list(_pipe);
    return _decodeMjs.map(_pipe$1, _resultMjs.values);
}
function on_submit(msg) {
    let _pipe = on("submit", _decodeMjs.subfield((0, _gleamMjs.toList)([
        "detail",
        "formData"
    ]), formdata_decoder(), (formdata)=>{
        let _pipe = formdata;
        let _pipe$1 = msg(_pipe);
        return _decodeMjs.success(_pipe$1);
    }));
    return prevent_default(_pipe);
}
function on_focus(msg) {
    return on("focus", _decodeMjs.success(msg));
}
function on_blur(msg) {
    return on("blur", _decodeMjs.success(msg));
}

},{"../../gleam_json/gleam/json.mjs":"8Pq32","../../gleam_stdlib/gleam/dynamic/decode.mjs":"gmHd7","../../gleam_stdlib/gleam/int.mjs":"32hLf","../../gleam_stdlib/gleam/pair.mjs":"5ZTSQ","../../gleam_stdlib/gleam/result.mjs":"oBmFG","../gleam.mjs":"jNPQG","../lustre/attribute.mjs":"faRXj","../lustre/effect.mjs":"iAEPi","../lustre/internals/constants.mjs":"gKFR6","../lustre/vdom/vattr.mjs":"jrrcC","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"5ZTSQ":[function(require,module,exports,__globalThis) {
/**
 * Returns the first element in a pair.
 *
 * ## Examples
 *
 * ```gleam
 * assert first(#(1, 2)) == 1
 * ```
 */ var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "first", ()=>first);
/**
 * Returns the second element in a pair.
 *
 * ## Examples
 *
 * ```gleam
 * assert second(#(1, 2)) == 2
 * ```
 */ parcelHelpers.export(exports, "second", ()=>second);
/**
 * Returns a new pair with the elements swapped.
 *
 * ## Examples
 *
 * ```gleam
 * assert swap(#(1, 2)) == #(2, 1)
 * ```
 */ parcelHelpers.export(exports, "swap", ()=>swap);
/**
 * Returns a new pair with the first element having had `with` applied to
 * it.
 *
 * ## Examples
 *
 * ```gleam
 * assert #(1, 2) |> map_first(fn(n) { n * 2 }) == #(2, 2)
 * ```
 */ parcelHelpers.export(exports, "map_first", ()=>map_first);
/**
 * Returns a new pair with the second element having had `with` applied to
 * it.
 *
 * ## Examples
 *
 * ```gleam
 * assert #(1, 2) |> map_second(fn(n) { n * 2 }) == #(1, 4)
 * ```
 */ parcelHelpers.export(exports, "map_second", ()=>map_second);
/**
 * Returns a new pair with the given elements. This can also be done using the dedicated
 * syntax instead: `new(1, 2) == #(1, 2)`.
 *
 * ## Examples
 *
 * ```gleam
 * assert new(1, 2) == #(1, 2)
 * ```
 */ parcelHelpers.export(exports, "new$", ()=>new$);
function first(pair) {
    let a;
    a = pair[0];
    return a;
}
function second(pair) {
    let a;
    a = pair[1];
    return a;
}
function swap(pair) {
    let a;
    let b;
    a = pair[0];
    b = pair[1];
    return [
        b,
        a
    ];
}
function map_first(pair, fun) {
    let a;
    let b;
    a = pair[0];
    b = pair[1];
    return [
        fun(a),
        b
    ];
}
function map_second(pair, fun) {
    let a;
    let b;
    a = pair[0];
    b = pair[1];
    return [
        a,
        fun(b)
    ];
}
function new$(first, second) {
    return [
        first,
        second
    ];
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"eHhmy":[function(require,module,exports,__globalThis) {
// Global state for the audio pipeline
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
parcelHelpers.export(exports, "toggle_audio_state", ()=>toggle_audio_state);
parcelHelpers.export(exports, "init_cartesia_stream", ()=>init_cartesia_stream);
let audioCtx = null;
let nextStartTime = 0;
const SAMPLE_RATE = 44100;
function toggle_audio_state(should_play) {
    if (!audioCtx) return;
    // Resuming/Suspending the context is the most efficient way to pause 
    // a scheduled timeline without losing our place.
    should_play ? audioCtx.resume() : audioCtx.suspend();
}
function init_cartesia_stream(transcript, apiKey, dispatch) {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
    });
    // Crucial for iOS: resume must happen inside the user-gesture call stack
    audioCtx.resume();
    nextStartTime = audioCtx.currentTime;
    // Use the latest stable 2026 versioning
    const url = `wss://api.cartesia.ai/tts/websocket?api_key=${apiKey}&cartesia_version=2024-06-10`;
    const socket = new WebSocket(url);
    socket.onopen = ()=>{
        const request = {
            model_id: "sonic-3",
            transcript: transcript,
            language: "en",
            voice: {
                mode: "id",
                id: "79a36f69-74f1-4177-8547-0e6d5e7542d1"
            },
            output_format: {
                container: "raw",
                encoding: "pcm_s16le",
                sample_rate: SAMPLE_RATE
            }
        };
        socket.send(JSON.stringify(request));
    };
    socket.onmessage = (event)=>{
        const response = JSON.parse(event.data);
        if (response.type === "chunk" && response.data) handleAudioChunk(response.data);
        if (response.done) {
            socket.close();
            dispatch({
                type: "HandleFFIEvent",
                data: "AudioEnded"
            });
        }
    };
    socket.onerror = (err)=>{
        console.error("Cartesia Error:", err);
        dispatch({
            type: "HandleFFIEvent",
            data: "SocketError"
        });
    };
}
function handleAudioChunk(base64Data) {
    const binaryString = window.atob(base64Data);
    const len = binaryString.length;
    const bytes = new Int16Array(len / 2);
    for(let i = 0; i < len; i += 2)bytes[i / 2] = binaryString.charCodeAt(i + 1) << 8 | binaryString.charCodeAt(i);
    const float32Data = new Float32Array(bytes.length);
    for(let i = 0; i < bytes.length; i++)float32Data[i] = bytes[i] / 32768.0;
    const buffer = audioCtx.createBuffer(1, float32Data.length, SAMPLE_RATE);
    buffer.copyToChannel(float32Data, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    // Drifting protection: If nextStartTime is too far in the past, 
    // reset it to the current context time to avoid "catch-up" bursts.
    const lookahead = 0.1; // 100ms safety buffer
    if (nextStartTime < audioCtx.currentTime) nextStartTime = audioCtx.currentTime + lookahead;
    source.start(nextStartTime);
    nextStartTime += buffer.duration;
}

},{"@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}],"aBxRS":[function(require,module,exports,__globalThis) {
var parcelHelpers = require("@parcel/transformer-js/src/esmodule-helpers.js");
parcelHelpers.defineInteropFlag(exports);
var _preludeMjs = require("../prelude.mjs");
parcelHelpers.exportAll(_preludeMjs, exports);

},{"../prelude.mjs":"ib0cp","@parcel/transformer-js/src/esmodule-helpers.js":"jnFvT"}]},["aZbtf","9GtLI"], "9GtLI", "parcelRequireb87e", {})

//# sourceMappingURL=static.76cf1a40.js.map
