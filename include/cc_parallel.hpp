// Пул потоков для CPU-части: особи популяции, строки матрицы, классы вершин графа по тегам.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cc {

// Число потоков машины; hardware_concurrency вправе вернуть 0.
inline int HardwareThreads() { return std::max(1, (int)std::thread::hardware_concurrency()); }

// Элементы раздаются атомарным счётчиком, а не равными диапазонами: работа на элемент разная (спуск у особей сходится
// за разное число ходов), и при статическом разбиении часть потоков простаивала бы. Потоки создаются один раз на пул,
// а вызывающий работает наравне с ними, поэтому их workers - 1.
class WorkerPool {
 public:
  using Body = std::function<void(int item, int worker)>;  // worker из [0, Workers())

  explicit WorkerPool(int workers) {
    for (int worker = 1; worker < workers; ++worker) threads_.emplace_back([this, worker] { Loop(worker); });
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    start_.notify_all();
    for (std::thread& thread : threads_) thread.join();
  }

  int Workers() const { return (int)threads_.size() + 1; }

  // Выполняет body для элементов [0, count) и возвращается, когда отработали все.
  void Run(int count, const Body& body) {
    if (threads_.empty()) {
      for (int item = 0; item < count; ++item) body(item, 0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      body_ = &body;
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      busy_ = (int)threads_.size();
      ++generation_;
    }
    start_.notify_all();
    Drain(0);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return busy_ == 0; });
  }

 private:
  void Drain(int worker) {
    for (int item = next_.fetch_add(1, std::memory_order_relaxed); item < count_;
         item = next_.fetch_add(1, std::memory_order_relaxed)) {
      (*body_)(item, worker);
    }
  }

  void Loop(int worker) {
    unsigned seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      start_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      lock.unlock();
      Drain(worker);
      lock.lock();
      if (--busy_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  const Body* body_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  unsigned generation_ = 0;
  int busy_ = 0;
  bool stop_ = false;
};

}  // namespace cc
