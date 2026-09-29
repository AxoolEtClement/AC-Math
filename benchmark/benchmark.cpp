#include <iostream>
#include <chrono>
#include <functional>
#include <utility>

namespace Benchmark {

    template <typename Func, typename... Args>
    void measure(int iterations, Func&& func, Args&&... args) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

		std::cout << "Total time for " << iterations << " iterations: " << elapsed.count() << " ms\n" 
			<< "Mean time for single iteration : " << elapsed.count() / iterations << " ms\n";
    }
}