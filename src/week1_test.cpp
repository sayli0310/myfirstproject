// week1_test.cpp
// Verifies: construction, bounds checking, matmul shape rule,
// matmul correctness, add, transpose, error handling.
//
// Compile:  g++ -std=c++17 -O2 -Iinclude src/week1_test.cpp -o week1_test
// Run:      ./week1_test

#include "../include/matrix.hpp"
#include <cassert>
#include <iostream>

void test_construction() {
    Matrix m(3, 4);
    assert(m.rows == 3 && m.cols == 4);
    std::cout << "[PASS] construction\n";
}

void test_bad_construction_throws() {
    bool threw = false;
    try {
        Matrix bad(-1, 5);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] bad construction caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_bounds_checking() {
    Matrix m(2, 2);
    bool threw = false;
    try {
        m.at(5, 5) = 1.0f;
    } catch (const std::out_of_range& e) {
        threw = true;
        std::cout << "[PASS] bounds check caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_matmul_correctness() {
    // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] . [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    Matrix a(2, 2);
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Matrix b(2, 2);
    b.at(0,0)=5; b.at(0,1)=6; b.at(1,0)=7; b.at(1,1)=8;
    Matrix c = a.matmul(b);
    assert(c.at(0,0) == 19 && c.at(0,1) == 22);
    assert(c.at(1,0) == 43 && c.at(1,1) == 50);
    std::cout << "[PASS] matmul correctness\n";
}

void test_matmul_shape_mismatch_throws() {
    Matrix a(2, 3);
    Matrix b(4, 2); // 3 != 4 -> should throw
    bool threw = false;
    try {
        Matrix c = a.matmul(b);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] matmul shape mismatch caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_add_and_transpose() {
    Matrix a(2, 2);
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Matrix sum = a.add(a);
    assert(sum.at(1,1) == 8);

    Matrix t = a.transpose();
    assert(t.at(0,1) == 3); // a.at(1,0)
    std::cout << "[PASS] add + transpose\n";
}

void test_dot_product() {
    std::vector<float> v1 = {1, 2, 3};
    std::vector<float> v2 = {4, 5, 6};
    float d = dot(v1, v2); // 1*4 + 2*5 + 3*6 = 32
    assert(d == 32.0f);
    std::cout << "[PASS] dot product\n";
}

int main() {
    std::cout << "=== Week 1: Matrix class tests ===\n\n";
    try {
        test_construction();
        test_bad_construction_throws();
        test_bounds_checking();
        test_matmul_correctness();
        test_matmul_shape_mismatch_throws();
        test_add_and_transpose();
        test_dot_product();
    } catch (const std::exception& e) {
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll week 1 tests passed.\n";
    return 0;
}
