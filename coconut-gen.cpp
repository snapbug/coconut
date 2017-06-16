#include "avro/Compiler.hh"
#include "avro/DataFile.hh"
#include "avro/Decoder.hh"
#include "avro/Encoder.hh"
#include "avro/ValidSchema.hh"
#include "weights.hxx"
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tuple>

std::vector<std::vector<double>> readCSV(const char *fname) {
	using namespace boost;
	std::ifstream inp(fname);
	std::vector<std::vector<double>> mtx;
	std::string line;

	while (getline(inp, line)) {
		std::vector<double> dblline;
		for (auto &i : tokenizer<escaped_list_separator<char>>(line))
			dblline.push_back(std::stod(i));
		mtx.push_back(dblline);
	}
	return mtx;
}

int main(void) {
	std::ifstream ifs("../weights.avsc");
	avro::ValidSchema weightScheme;
	avro::compileJsonSchema(ifs, weightScheme);
	avro::DataFileReader<donk::nnweights> dfr("../weights.avro");
	std::vector<donk::nnweights> weights;
	donk::nnweights tmp;
	while (dfr.read(tmp))
		weights.push_back(tmp);

	std::vector<std::string> names{
	    "question_convolution_filters", "question_convolution_biases", "answer_convolution_filters",
	    "answer_convolution_biases",    "hidden_layer_weights",        "hidden_layer_biases",
	    "softmax_layer_weights",        "softmax_layer_biases",
	};

	std::ofstream coconut("coconut.cpp");
	coconut << R"(
#include <blaze/Math.h>
#include <iostream>
#include <vector>

using namespace blaze;

int main(void) {
)";
	for (int i = 0; i < weights.size(); i++) {
		const auto &weight = weights[i];
		const auto &name = names[i];
		double sz;

		switch (weight.dimension.size()) {
		default:
			std::cout << "Cannot handle > 3 dimensions!\n" << std::endl;
			continue;
			break;
		case 3:
			coconut << "std::vector<"
			        << "StaticMatrix<double, " << weight.dimension[1] << ", " << weight.dimension[2]
			        << ">> " << name << "_v;";
			coconut << name << "_v.reserve(" << weight.dimension[0] << ");\n";
			sz = weight.dimension[1] * weight.dimension[2];
			for (int n = 0; n < weight.dimension[0]; n++) {
				coconut << name << "_v.push_back(StaticMatrix<double," << weight.dimension[1] << ","
				        << weight.dimension[2] << ">{\n";
				auto start = n * sz;
				auto end = (n + 1) * sz;
				/* std::cout << "Weights from " << start << " to " << end << std::endl;; */
				coconut << "{";
				/* std::cout << "{"; */
				for (int w = start; w < end; w++) {
					if (w != start && w % weight.dimension[2] == 0) {
						coconut << "},\n{";
						/* std::cout << "},\n{"; */
					}
					coconut << weight.weights[w] << ", ";
					/* std::cout << weight.weights[w] << ", "; */
				}
				coconut << "}\n});\n";
				/* std::cout << "}\n});\n"; */
			}
			break;
		case 2:
			std::cerr << name << std::endl;
			coconut << "StaticMatrix<double, " << weight.dimension[0] << ", " << weight.dimension[1]
			        << "> " << name << "{\n";
			sz = weight.dimension[1];
			for (int x = 0; x < weight.dimension[0]; x++) {
				auto start = x * sz;
				auto end = (x + 1) * sz;
				coconut << "{";
				for (int w = start; w < end; w++)
					coconut << weight.weights[w] << ", ";
				coconut << "},\n";
			}
			coconut << "};\n";
			break;
		case 1:
			coconut << "StaticVector<double, " << weight.dimension[0] << "> " << name << "{\n";
			for (auto w : weight.weights) {
				coconut << w << ", ";
			}
			coconut << "};\n";
			break;
		}
	}
	/*
	 * The hidden layer and softmax layer weights were exported transposed, so transpose them to get
	 * the right ones
	 */
	for (auto fix : {6}) {
		coconut << "StaticMatrix<double, " << weights[fix].dimension[1] << ", "
		        << weights[fix].dimension[0] << "> fixed_" << names[fix] << " = trans("
		        << names[fix] << ");\n";
	}
	coconut << "transpose(hidden_layer_weights);\n";

	/*
	 * For now, hard code the question/answer sections
	 */
	auto question = readCSV("../q1.csv");
	auto answer = readCSV("../a1.csv");
	auto external = readCSV("../ext_1.csv");

	/*
	 * Add padding to question and answer, the hardcoded 4 is eww, but meh
	 */
	coconut << "StaticMatrix<double," << question.size() << "," << (question[1].size() + 8)
	        << "> question{\n";
	for (auto x : question) {
		coconut << "{0,0,0,0,";
		for (auto w : x)
			coconut << w << ", ";
		coconut << "0,0,0,0},\n";
	}
	coconut << "};\n";
	coconut << "StaticMatrix<double," << answer.size() << "," << (answer[0].size() + 8)
	        << "> answer{\n";
	for (auto x : answer) {
		coconut << "{0,0,0,0,";
		for (auto w : x)
			coconut << w << ", ";
		coconut << "0,0,0,0},\n";
	}
	coconut << "};\n";
	coconut << "StaticVector<double," << external.size() << "> external{\n";
	for (int i = 0; i < external.size(); i++) {
		coconut << external[i][0] << ",";
	}
	coconut << "\n};\n";

	coconut << R"(
auto performConvolution = [](auto input, auto filter) {
/*
 * - 8 because of the 4 padding on either side
 */
  StaticVector<double, input.columns() - 4> result;
  for (int i = 0; i < result.size(); i++) {
    auto dd = submatrix(input, 0, i, filter.rows(), filter.columns());
    auto cc = dd % filter;
    double sum = 0;
    for (int j = 0; j < cc.rows(); j++) {
      for (auto it = cc.begin(j); it != cc.end(j); ++it) {
        sum += *it;
      }
    }
    result[i] = sum;
  }
  return result;
};

StaticVector<double, question_convolution_biases.size()> question_conv_map;
for (int i = 0; i < question_convolution_filters_v.size(); i++) {
  question_conv_map[i] = max(performConvolution(question, question_convolution_filters_v[i]));
}
question_conv_map += question_convolution_biases;
question_conv_map = tanh(question_conv_map);

StaticVector<double, answer_convolution_biases.size()> answer_conv_map;
for (int i = 0; i < answer_convolution_filters_v.size(); i++) {
  answer_conv_map[i] = max(performConvolution(answer, answer_convolution_filters_v[i]));
}
answer_conv_map += answer_convolution_biases;
answer_conv_map = tanh(answer_conv_map);

StaticVector<double, question_convolution_biases.size() + answer_convolution_biases.size() + external.size()> joinLayer;
subvector(joinLayer, 0, question_conv_map.size()) = question_conv_map;
subvector(joinLayer, question_conv_map.size(), answer_conv_map.size()) = answer_conv_map;
subvector(joinLayer, question_conv_map.size() + answer_conv_map.size(), external.size()) = external;

/*
 * Have to evaluate() to get the types to match up correctly, not having it causes all sorts of template errors
 */
auto HiddenLayer = tanh(evaluate(trans(joinLayer) * hidden_layer_weights) + trans(hidden_layer_biases)) * 2;

StaticMatrix<double, softmax_layer_weights.columns(), softmax_layer_weights.rows()> softmax_layer_weights_fix = trans(softmax_layer_weights);

auto FinalLayer = trans(evaluate(evaluate(HiddenLayer * softmax_layer_weights_fix) + trans(softmax_layer_biases)));

StaticVector<double, 2> fmax(max(FinalLayer));
auto submax = FinalLayer - fmax;
auto expsubmax = evaluate(exp(submax));
auto sumexpsubmax = expsubmax[0] + expsubmax[1];
std::cout << submax[0] - log(sumexpsubmax) << ", " << submax[1] - log(sumexpsubmax) << std::endl;
)";

	coconut << "}\n";
	return EXIT_SUCCESS;
}
