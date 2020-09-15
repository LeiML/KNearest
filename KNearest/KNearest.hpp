//
// Created by LeiLei on 2020/9/14.
//

#ifndef KNearest_HPP
#define KNearest_HPP
#include <Matrix.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <string>
#include <regex>
#include <map>
#include <set>
#define COLS 1025

using namespace std;
using namespace filesystem;

std::vector<std::string>files(const std::string& directory);

string text(const string& file);

vector<int>lines(const string& line);

void mnist(Matrix& data, Matrix& label, const string&path);

vector<int>digital(const vector<string>&path);


class KNearest{
public:
    KNearest(const Matrix& Sample, const Matrix& label, const int& k = 3);
    Matrix predict(const Matrix& test);
private:
    Matrix sample;
    Matrix label;
    int k;
protected:
    Matrix distance(const Matrix&test);
    Matrix index(const Matrix& distance);
};

template<typename Tp>
map<Tp, int>count(const vector<Tp>&data){
    map<Tp, int>result;
    set<Tp> keys = set<Tp>(data.begin(), data.end());
    for (auto it=keys.begin();it!=keys.end();it++){
        int num = 0;
        for (int i = 0; i < data.size(); ++i) {
            if (*it == data.at(i))
                num++;
        }
        result.insert(make_pair(*it, num));
    }
    return result;
}

#endif //KNearest_HPP
