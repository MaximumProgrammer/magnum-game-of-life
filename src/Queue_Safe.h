/* Markus Gruber
 * Markus.Gruber4@gmx.net
 */
#include <queue>
#include <condition_variable>
#include <mutex>

template <typename T>
class Queue_Safe {
private:
    std::queue<T> q;               // Underlying queue to store elements
    std::condition_variable cv;    // Condition variable for synchronization
    std::mutex mtx;                // Mutex for exclusive access to the queue

public:
    Queue_Safe() {}

    // Pushes an element onto the queue
    void push(T const& val) {
      std::lock_guard<std::mutex> lock(mtx);
      q.push(val);
      cv.notify_one();  // Notify one waiting thread that data is available
    }

    // Pops and returns the front element of the queue
    T pop()
    {
      std::unique_lock<std::mutex> uLock(mtx);
      cv.wait(uLock,
              [&] { return !q.empty(); });  // Wait until the queue is not empty
      T front = q.front();
      q.pop();
      return front;
    }
    bool empty() { return !q.size(); }
    auto size() { return q.size(); }
};
