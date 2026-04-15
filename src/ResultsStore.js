/**
 * ResultsStore — in-memory singleton for the last /api/analyze result.
 * Lives in JS module scope: survives React Router navigation (no page reload)
 * without any localStorage/sessionStorage quota constraints.
 * Cleared on hard refresh (expected — user must re-upload).
 */
let _result = null;

export const setLastResult = (r) => { _result = r; };
export const getLastResult = () => _result;
export const clearLastResult = () => { _result = null; };
