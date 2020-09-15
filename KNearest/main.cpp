#include "KNearest.hpp"

int main() {
    Matrix train_x, train_y, test_x, test_y;
    mnist(train_x, train_y, R"(E:\BasicProgram\Source\mnist\train)");
    mnist(test_x, test_y, R"(E:\BasicProgram\Source\mnist\test)");
    KNearest nearest = KNearest(train_x, train_y);
    Matrix result = nearest.predict(test_x);
    cout << "the true label: \t" << test_y << endl;
    cout << "the predict label: \t" << result << endl;
    return 0;
}
