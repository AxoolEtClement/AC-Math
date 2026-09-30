#include <nanobench.h>
#include "mathLibCPP/Vec3.h"

int main() {
	Maths::Vec3<float> v1{ 1.0f, 2.0f, 3.0f };
	Maths::Vec3<float> v2{ 4.0f, 5.0f, 6.0f };
	ankerl::nanobench::Bench().run("Dot product vec3", [&] {
		v1.Dot(v2);
	});

	return 0;
}