//
// Created by LeiLei on 2020/9/14.
//
#include "KNearest.hpp"

std::vector<std::string>files(const std::string& directory){
    std::vector<std::string>files;
    path path(directory);
    assert(is_directory(path) || exists(path));
    for (const auto& file: directory_iterator(path))
        files.push_back(file.path().string());
    return files;
}

string text(const string& file){
    string text;
    assert(is_regular_file(path(file)));
    fstream fs;
    fs.open(file);
    assert(fs.is_open());
    char c;
    while(!fs.eof()){
        fs >> c;
        text += c;
    }
    return text;
}

vector<int>lines(const string& line){
    vector<int>data;
    for(const char& c : line)
        data.push_back(std::stoi(string(1, c)));
    return data;
}

vector<int>digital(const vector<string>&path){
    vector<int>digital;
    regex pattern("\\d+");
    smatch result;
    for(const string&str: path){
        regex_search(str.begin(), str.end(), result, pattern);
        digital.push_back(std::stoi(result[0]));
    }
    return digital;
}

void mnist(Matrix& data, Matrix& label, const string&path){
    vector<string>file = files(path);
    vector<int>digit = digital(file);
    label = Matrix(digit.size(), 1, digit);
    vector<int>mat;
    for (const auto & f : file) {
        vector<int>line = lines(text(f));
        mat.insert(mat.end(), line.begin(), line.end());
    }
    data = Matrix(digit.size(), COLS, mat);
}

KNearest::KNearest(const Matrix &Sample, const Matrix &Label, const int &k) : sample(Sample), label(Label), k(k){}

Matrix KNearest::predict(const Matrix &test) {
    Matrix distant = this->distance(test);
    Matrix indexes = index(distant);
    Matrix temp = Matrix(indexes.size());
    for (int i = 0; i < test.row; ++i)
        for (int j = 0; j < this->k; ++j)
            temp.at(i, j) = this->label.at(int(indexes.at(i, j)), 0);
    Matrix result = Matrix(test.row, 1);
    for (int i = 0; i < test.row; ++i) {
        map<int, int> data = count<int>(temp.rows(i).vec<int>());
        int max = 0;
        for (const auto & it : data) {
            if (max < it.second) {
                max = it.second;
                result.at(i, 0) = it.first;
            }
        }
    }
    return result;
}

Matrix KNearest::distance(const Matrix&test) {
    assert(test.col == this->sample.col);
    Matrix result = Matrix(test.row, this->sample.row);
    for (int i = 0; i < test.row; ++i) {
        for (int j = 0; j < this->sample.row; ++j) {
            result.at(i, j) = std::sqrt(pow(test.rows(i) - this->sample.rows(j), 2).sum());
        }
        cout << "the data have been trained: " << i+1 << ", the total data: " << test.row << endl;
    }
    return result;
}

Matrix KNearest::index(const Matrix &distance) {
    Matrix result = Matrix(distance.row, this->k);
    for (int i = 0; i < distance.row; ++i) {
        vector<double>data = distance.rows(i).vec<double>();
        vector<double>temp = data;
        sort(temp.begin(), temp.end());
        for (int j = 0; j < this->k; ++j) {
            auto it = find(data.begin(), data.end(), temp.at(j));
            result.at(i, j) = std::distance(std::begin(data), it);
        }
    }
    return result;
}
