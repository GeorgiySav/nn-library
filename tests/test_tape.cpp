#include "test_harness.h"

#include <nn/ops/ops.h>
#include <nn/autograd/tape.h>

nn::Tensor make_param() {
  nn::Tensor p = nn::Tensor::scalar(0.0f);
  p.set_requires_grad(true);
  return p;
}

nn::autograd::BackwardFn contributes(float v) {
  return [v](const nn::Tensor&, std::span<nn::Tensor> g_in) {
    for (nn::Tensor& g : g_in) g = nn::Tensor::scalar(v);
  };
}

void times_two(const nn::Tensor& g_out, std::span<nn::Tensor> g_in) {
  for (nn::Tensor& g : g_in) g = nn::Tensor::scalar(2.0f * g_out.item());
}

void seed_from(nn::Tensor& t, int id) {
  t.ensure_meta().node_id = id;
}

NN_TEST(test_active_tape) {
  NN_CHECK(nn::autograd::active_tape() == nullptr);
  {
    nn::autograd::Tape tape;
    nn::autograd::TapeScope scope(tape);
    NN_CHECK(nn::autograd::active_tape() == &tape);
    NN_CHECK(nn::autograd::grad_enabled() == true);
  }
  NN_CHECK(nn::autograd::active_tape() == nullptr);
}

NN_TEST(test_no_grad_scope) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);
  NN_CHECK(nn::autograd::grad_enabled() == true);
  {
    nn::autograd::NoGradScope no_grad_scope;
    NN_CHECK(nn::autograd::grad_enabled() == false);
    {
      nn::autograd::NoGradScope no_grad_scope2;
      NN_CHECK(nn::autograd::grad_enabled() == false);
    }
    NN_CHECK(nn::autograd::grad_enabled() == false);
  }
  NN_CHECK(nn::autograd::grad_enabled() == true);
}

NN_TEST(test_linear_chain) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf}, "x2a");
  const int n1 = tape.record(times_two, {n0}, "x2b");
  const int n2 = tape.record(times_two, {n1}, "x2c");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(loss, n2);
  tape.backward(loss);

  NN_CHECK_CLOSE(param.grad().item(), 8.0f, 1e-6);
}

NN_TEST(test_accumulates_fan_out) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);
  
  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int b = tape.record(contributes(3.0f), {leaf}, "b");
  const int c = tape.record(contributes(3.0f), {leaf}, "c");
  const int d = tape.record(contributes(1.0f), {b, c}, "join");
  
  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(loss, d);
  tape.backward(loss);
  
  NN_CHECK_CLOSE(param.grad().item(), 6.0f, 1e-6);
}

NN_TEST(test_leaf_accumulates_across_backwards) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(loss, n0);

  tape.backward(loss);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);

  tape.backward(loss);
  NN_CHECK_CLOSE(param.grad().item(), 4.0f, 1e-6);

  param.zero_grad();
  tape.backward(loss);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);
}
NN_TEST(test_tape_skips_untracked_inputs) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf, -1}, "one_tracked");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(loss, n0);
  tape.backward(loss);

  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);
}

NN_TEST(test_tape_skips_unreached_nodes) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  bool orphan_ran = false;
  tape.record([&orphan_ran](const nn::Tensor&, std::span<nn::Tensor>) {
    orphan_ran = true;
  }, {}, "orphan");

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(loss, n0);
  tape.backward(loss);

  NN_CHECK(orphan_ran == false);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);
}

NN_TEST(tape_backward_reject_non_scalar_loss) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor not_scalar = nn::Tensor::zeros({2, 2});
  seed_from(not_scalar, n0);

  NN_CHECK_THROWS(tape.backward(not_scalar), std::invalid_argument);
}

NN_TEST(tape_clear_resets_size) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.record_leaf(&param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  NN_CHECK(tape.size() == 2);
  tape.clear();
  NN_CHECK(tape.size() == 0);
}

NN_TEST(tape_record_leaf_is_idempotent) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf1 = tape.record_leaf(&param);
  const int leaf2 = tape.record_leaf(&param);

  NN_CHECK(leaf1 == leaf2);
  NN_CHECK(tape.size() == 1);
}