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

void seed_from(nn::autograd::Tape& tape, nn::Tensor& t, int id) {
  tape.set_producer(t, id);
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
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf}, "x2a");
  const int n1 = tape.record(times_two, {n0}, "x2b");
  const int n2 = tape.record(times_two, {n1}, "x2c");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n2);
  tape.backward(loss);

  NN_CHECK_CLOSE(param.grad().item(), 8.0f, 1e-6);
}

NN_TEST(test_accumulates_fan_out) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);
  
  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int b = tape.record(contributes(3.0f), {leaf}, "b");
  const int c = tape.record(contributes(3.0f), {leaf}, "c");
  const int d = tape.record(contributes(1.0f), {b, c}, "join");
  
  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, d);
  tape.backward(loss);
  
  NN_CHECK_CLOSE(param.grad().item(), 6.0f, 1e-6);
}

NN_TEST(test_leaf_accumulates_across_backwards) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n0);

  tape.backward(loss, true);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);

  tape.backward(loss, true);
  NN_CHECK_CLOSE(param.grad().item(), 4.0f, 1e-6);

  param.zero_grad();
  tape.backward(loss);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);
}
NN_TEST(test_tape_skips_untracked_inputs) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf, -1}, "one_tracked");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n0);
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
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n0);
  tape.backward(loss);

  NN_CHECK(orphan_ran == false);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);
}

NN_TEST(tape_backward_reject_non_scalar_loss) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  nn::Tensor not_scalar = nn::Tensor::zeros({2, 2});
  seed_from(tape, not_scalar, n0);

  NN_CHECK_THROWS(tape.backward(not_scalar), std::invalid_argument);
}

NN_TEST(tape_clear_resets_size) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0 = tape.record(times_two, {leaf}, "x2");

  NN_CHECK(tape.size() == 2);
  tape.clear();
  NN_CHECK(tape.size() == 0);
}

NN_TEST(tape_node_for_is_idempotent) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  NN_CHECK(tape.node_for(param) == tape.node_for(param));
  NN_CHECK(tape.size() == 1);
}

NN_TEST(tape_node_for_reuses_op_outputs) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int op   = tape.record(times_two, {leaf}, "x2");

  nn::Tensor out = nn::Tensor::scalar(0.0f);
  tape.set_producer(out, op);

  NN_CHECK(tape.node_for(out) == op);
  NN_CHECK(tape.size() == 2);
}

NN_TEST(tape_node_for_skips_untracked_tensors) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor plain = nn::Tensor::scalar(1.0f);   // no meta at all
  NN_CHECK(tape.node_for(plain) == -1);

  nn::Tensor frozen = nn::Tensor::scalar(1.0f);
  frozen.set_requires_grad(false);               // has meta, but frozen
  NN_CHECK(tape.node_for(frozen) == -1);

  NN_CHECK(tape.size() == 0);                    // neither grew the tape
}

NN_TEST(tape_clear_invalidates_stamped_ids) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  tape.record(times_two, {}, "filler");
  nn::Tensor param = make_param();
  NN_CHECK(tape.node_for(param) == 1);

  tape.clear();

  // param still carries node_id 1 from the old epoch; it must be ignored.
  NN_CHECK(tape.node_for(param) == 0);
  NN_CHECK(tape.size() == 1);
}

NN_TEST(tape_backward_rejects_stale_loss) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0   = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n0);
  tape.clear();

  // Before the epoch, this indexed grads_[1] on an empty vector.
  NN_CHECK_THROWS(tape.backward(loss), std::invalid_argument);
}

NN_TEST(tape_backward_clears_by_default) {
  nn::autograd::Tape tape;
  nn::autograd::TapeScope scope(tape);

  nn::Tensor param = make_param();
  const int leaf = tape.node_for(param);
  const int n0   = tape.record(times_two, {leaf}, "x2");

  nn::Tensor loss = nn::Tensor::scalar(0.0f);
  seed_from(tape, loss, n0);

  tape.backward(loss);
  NN_CHECK(tape.size() == 0);
  NN_CHECK_CLOSE(param.grad().item(), 2.0f, 1e-6);  // grad survives the clear
}

NN_TEST(tape_scope_rejects_a_second_tape) {
  nn::autograd::Tape a, b;
  nn::autograd::TapeScope scope(a);

  {   // re-entering the same tape is fine
    nn::autograd::TapeScope inner(a);
    NN_CHECK(nn::autograd::active_tape() == &a);
  }

  NN_CHECK_THROWS(nn::autograd::TapeScope{b}, std::logic_error);
  NN_CHECK(nn::autograd::active_tape() == &a);   // failed ctor left it alone
}
