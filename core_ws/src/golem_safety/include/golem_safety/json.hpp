#pragma once

#include <string>

namespace golem_safety
{

// `text` as a JSON string literal, quotes included: the quote and the backslash are
// backslash-escaped, a newline is written as backslash and n, and every other control
// character below 0x20 as backslash, u and four lowercase hex digits. Bytes from 0x20 up
// pass through unchanged.
std::string json_string(const std::string & text);

}  // namespace golem_safety
