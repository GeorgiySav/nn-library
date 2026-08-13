#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nn::test {

struct TestCase { const char* name; void (*fn)(); };
inline std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; };
inline int& fail_count() { static int f = 0; return f; }
inline const char*& current() { static const char* c = ""; return c; }

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void report(const char* file, int line, const std::string& msg) {
  std::printf(" FAIL %s:%d in %s\n  %s\n", file, line, current(), msg.c_str());
  ++fail_count();
}

inline int run_all(int argc, char** argv) {
  const char* filter = (argc > 1) ? argv[1] : nullptr;
  int ran = 0, failed_tests = 0;

  for (const auto& tc : registry()) {
    if (filter && !std::strstr(tc.name, filter)) continue;
    current() = tc.name;
    int before = fail_count();
    std::printf("[ RUN ] %s\n", tc.name);
    tc.fn();
    bool ok = (fail_count() == before);
    if (!ok) ++failed_tests;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", tc.name);
    ++ran;
  }

  std::printf("\n%d test(s) run, %d failed, %d assertion failure(s)\n", ran, failed_tests, fail_count());
  return failed_tests == 0 ? 0 : 1;
}

} // namespace nn::test

#define NN_TEST(name) \
  static void name(); \
  static nn::test::Registrar registrar_##name(#name, name); \
  static void name()

#define NN_CHECK(cond) \
  do { \
    if (!(cond)) ::nn::test::report(__FILE__, __LINE__, "NN_CHECK(" #cond ")"); \
  } while (0)

#define NN_CHECK_CLOSE(a, b, tol) \
  do { \
    double a_ = (a), b_ = (b), t_ = (tol); \
    double d_ = std::fabs(a_ - b_); \
    double r_ = d_ / std::fmax(1e-8, std::fmax(std::fabs(a_), std::fabs(b_))); \
    if (r_ > t_ && d_ > t_)                                              \
      ::nn::test::report(__FILE__, __LINE__,                             \
        "NN_CHECK_CLOSE(" #a ", " #b "): " + std::to_string(a_) + " vs " \
        + std::to_string(b_) + " (rel " + std::to_string(r_) + ")");     \
  } while (0)