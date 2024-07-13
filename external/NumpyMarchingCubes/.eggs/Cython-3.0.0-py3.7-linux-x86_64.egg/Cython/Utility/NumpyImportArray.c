///////////////////////// NumpyImportArray.init ////////////////////

// comment below is deliberately kept in the generated C file to
// help users debug where this came from:
/*
 * Cython has automatically inserted a call to _import_array since
 * you didn't include one when you cimported numpy. To disable this
 * add the line
 *   <void>numpy._import_array
 */
#ifdef NPY_FEATURE_VERSION /* This is a public define that makes us reasonably confident it's "real" Numpy */
// NO_IMPORT_ARRAY is Numpy's mechanism for indicating that import_array is handled elsewhere
#if !NO_IMPORT_ARRAY /* https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.array.html#c.NO_IMPORT_ARRAY  */
if (unlikely(_import_array() == -1)) {
    PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import "
    "(auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy; "
    "use '<void>numpy._import_array' to disable if you are certain you don't need it).");
}
#endif
#endif

///////////////////////// NumpyImportUFunc.init ////////////////////

// Unlike import_array, this is generated by the @cython.ufunc decorator
// so we're confident the right headers are present and don't need to override them

{
    // NO_IMPORT_UFUNC is Numpy's mechanism for indicating that import_umath is handled elsewhere
#if !NO_IMPORT_UFUNC /* https://numpy.org/devdocs/reference/c-api/ufunc.html#c.NO_IMPORT_UFUNC */
    if (unlikely(_import_umath() == -1)) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.umath failed to import "
        "(auto-generated by @cython.ufunc).");
    }
#else
    if ((0)) {}
#endif
    // NO_IMPORT_ARRAY is Numpy's mechanism for indicating that import_array is handled elsewhere
#if !NO_IMPORT_ARRAY /* https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.array.html#c.NO_IMPORT_ARRAY  */
    else if (unlikely(_import_array() == -1)) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import "
        "(auto-generated by @cython.ufunc).");
    }
#endif
}


