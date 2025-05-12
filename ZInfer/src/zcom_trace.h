#ifndef __ZCOM_TRACE_H__
#define __ZCOM_TRACE_H__

#include <iostream>

#if 0
#define INFO_TRACE(str) std::cout << "I "<< str << std::endl
#define ERROR_TRACE(str) std::cout << "E "<< str << std::endl
#define INFO_TRACE_THIS(str) std::cout << "I "<< str << " this=" << this << std::endl
#define ERROR_TRACE_THIS(str) std::cout << "E "<< str << " this="  << this << std::endl
#define MASSERT_RETURN(exp, rv) \
{ \
    if (!(exp)) { \
        std::cout << __FILE__ << ": " << __LINE__ << ", assert failed, " << #exp << std::endl; \
        return rv; \
    } \
}

#define MASSERT_RETURN_VOID(exp) \
{ \
    if (!(exp)) { \
        std::cout << __FILE__ << ": " << __LINE__ << ", assert failed, " << #exp << std::endl; \
        return; \
    } \
}
#endif

constexpr char level_prefix[] = { 'D', 'I', 'W', 'E' };
enum MTRACE_LEVEL_TYPE
{
    MTRACE_LEVEL_DEBUG,
    MTRACE_LEVEL_INFO,
    MTRACE_LEVEL_WARNING,
    MTRACE_LEVEL_ERROR
};

inline const char* path_to_name(const char* path, char delim)
{
    const char* last_delim = strrchr(path, delim);
    if (last_delim && *last_delim)
    {
        return last_delim + 1;
    }
    return path;
}

#if defined(_WIN32) || defined(_WIN64)
#define __FILENAME__ path_to_name(__FILE__, '\\')
#else
#define __FILENAME__ path_to_name(__FILE__, '/')
#endif

#define __FUNCNAME_ONLY__ path_to_name(__FUNCTION__, ':')

/////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MTRACE(str, Level)  { \
                                std::cout << level_prefix[Level] << ": " << \
								__FILENAME__ << "(" << \
								__LINE__ << ") " << \
								__FUNCNAME_ONLY__ << "() " << \
								str << \
								std::endl; \
                            }

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// interfaces provided
#define DEBUG_TRACE(str)      MTRACE(str, MTRACE_LEVEL_DEBUG)
#define INFO_TRACE(str)       MTRACE(str, MTRACE_LEVEL_INFO)
#define WARNING_TRACE(str)    MTRACE(str, MTRACE_LEVEL_WARNING)
#define ERROR_TRACE(str)      MTRACE(str, MTRACE_LEVEL_ERROR)

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
// using inside non-static class members
#define DEBUG_TRACE_THIS(str)    DEBUG_TRACE(str << ", this=" << this)
#define INFO_TRACE_THIS(str)     INFO_TRACE(str << ", this=" << this)
#define WARNING_TRACE_THIS(str)  WARNING_TRACE(str << ", this=" << this)
#define ERROR_TRACE_THIS(str)    ERROR_TRACE(str << ", this=" << this)

#define ZASSERT_RETURN(exp, rv) \
{ \
    if (!(exp)) { \
        ERROR_TRACE("ztrace_assert: " << #exp); \
        return rv; \
    } \
}

#define ZASSERT_RETURN_VOID(exp) \
{ \
    if (!(exp)) { \
        ERROR_TRACE("ztrace_assert: " << #exp); \
        return; \
    } \
}

#define ZASSERT_RETURN_WITHSTR(exp, rv, str) \
{ \
    if (!(exp)) { \
        ERROR_TRACE("ztrace_assert: " << str); \
        return rv; \
    } \
}

#define ZASSERT_RETURN_VOID_WITHSTR(exp, str) \
{ \
    if (!(exp)) { \
        ERROR_TRACE("ztrace_assert: " << str); \
        return; \
    } \
}

#endif  // end of __ZCOM_TRACE_H__
