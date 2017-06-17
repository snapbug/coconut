#include "avro/Compiler.hh"
#include "avro/DataFile.hh"
#include "avro/Decoder.hh"
#include "nnweights.hxx"
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

int main(int argc, char **argv) {
	std::ofstream coconut("coconut.cpp");
	coconut << R"(
#include "blaze/Math.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

static constexpr unsigned long COLUMN_PADDING = 4;
static constexpr unsigned long MAX_SENTENCE_LENGTH = 60;
static constexpr unsigned long EMBED_DIMENSION = 50;

using namespace blaze;

auto loadWord2Vec(const char *fname) {
  using ValueType = StaticMatrix<float, EMBED_DIMENSION, 1>;
  std::ifstream w2v(fname, std::ios::binary);
  std::unordered_map<std::string, ValueType> w2vmap;
  long long vocab, layer;

  w2v >> vocab >> layer;
  if (layer > EMBED_DIMENSION) {
    std::cerr << "Embedding size (" << layer << ") is larger than allowed (" << EMBED_DIMENSION << ")\n";
    std::terminate();
  }
  std::cerr << "Loading " << EMBED_DIMENSION << "-dimensional embeddings for " << vocab << " terms: ";

  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < vocab; i++) {
    std::string word;
    std::vector<float> vec;
    vec.reserve(layer);

    w2v >> word;
    w2v.get(); // skip over space after word
    for (int j = 0; j < layer; j++) {
      float x;
      w2v.read((char *)&x, sizeof(x));
      vec.push_back(x);
    }
    w2vmap[word] = ValueType(EMBED_DIMENSION, 1, vec.data());
  }
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> time = end - start;
  std::cerr << time.count() << "s\n";
  return w2vmap;
}

int main(void) {
)";

	std::vector<std::string> names{
      "question_convolution_filters",
      "question_convolution_biases",
      "answer_convolution_filters",
      "answer_convolution_biases",
      "hidden_layer_weights",
      "hidden_layer_biases",
      "softmax_layer_weights",
      "softmax_layer_biases"
	};

	avro::DataFileReader<coconut::cnnweights> dfr(argv[1]);
	coconut::cnnweights weight;
	int i = 0;
	while (dfr.read(weight)) {
		const auto name = names[i++];
		double sz;

		switch (weight.dimension.size()) {
		default:
			std::cerr << "Cannot handle > 3 dimensions!\n" << std::endl;
			continue;
		case 3:
			coconut << "std::vector<" << "StaticMatrix<double, " << weight.dimension[1] << ", " << weight.dimension[2] << ">> " << name << ";\n";
			coconut << name << ".reserve(" << weight.dimension[0] << ");\n";
			sz = weight.dimension[1] * weight.dimension[2];
			for (int n = 0; n < weight.dimension[0]; n++) {
				coconut << name << ".push_back(StaticMatrix<double, " << weight.dimension[1] << ", " << weight.dimension[2] << ">{";
				auto start = n * sz;
				auto end = (n + 1) * sz;
				coconut << "{";
				for (int w = start; w < end; w++) {
					if (w != start && w % weight.dimension[2] == 0)
						coconut << "}, {";
					coconut << weight.weights[w] << ", ";
				}
				coconut << "}});\n";
			}
			break;
		case 2:
			coconut << "StaticMatrix<double, " << weight.dimension[0] << ", " << weight.dimension[1] << "> " << name << "{";
			sz = weight.dimension[1];
			for (int x = 0; x < weight.dimension[0]; x++) {
				auto start = x * sz;
				auto end = (x + 1) * sz;
				coconut << "{";
				for (int w = start; w < end; w++)
					coconut << weight.weights[w] << ", ";
				coconut << "},";
			}
			coconut << "};\n";
			break;
		case 1:
			coconut << "StaticVector<double, " << weight.dimension[0] << "> " << name << "{";
			for (auto w : weight.weights)
				coconut << w << ", ";
			coconut << "};\n";
			break;
		}
	}

	/*
	 * The hidden layer weights were exported transposed, so transpose them to get
	 * the right ones
	 */
	coconut << "transpose(hidden_layer_weights);\n";

	/*
	 * Generate a "random" vector for unknown words -- while this _should_ be random I'm too dumb to
	 * work out how to use the same random number generator as numpy, so I borrowed this from the
	 * output of the weights for an example question...
	 */
	coconut << R"(
StaticMatrix<float, EMBED_DIMENSION, 1> unknown_question_word{{0.120687},{-0.0994655},{-0.0388413},{0.340653},{-0.047662},{-0.585873},{0.302925},{0.493592},{0.034791},{-0.165766},{0.395165},{0.135953},{-0.443345},{-0.274928},{0.355806},{0.614136},{-0.267557},{-0.0167247},{0.41984},{-0.237471},{0.742444},{-0.314932},{0.0132752},{-0.36691},{-0.265743},{-0.524515},{0.122545},{-0.256592},{0.176823},{-0.402641},{-0.265591},{-0.206455},{-0.285454},{0.276018},{0.474488},{0.28826},{-0.0081898},{0.0514516},{0.0654361},{0.21053},{-0.20768},{-0.0979844},{0.0758915},{0.348158},{0.635769},{0.258099},{-0.119316},{0.92223},{0.0483396},{-0.123451}};
StaticMatrix<float, EMBED_DIMENSION, 1> unknown_answer_word{{-0.15424},{0.0610544},{-0.0311361},{0.142679},{0.139988},{-0.113704},{-0.111768},{0.150936},{0.22907},{0.187966},{-0.0710914},{0.000497563},{0.0917315},{0.106351},{-0.0648746},{0.0305981},{0.00154158},{-0.243116},{0.136413},{0.191321},{-0.067557},{0.0576981},{-0.212309},{-0.065588},{0.21657},{0.0756891},{-0.0513987},{0.144365},{-0.0915819},{0.0340493},{0.184564},{-0.0319133},{0.151074},{-0.178117},{0.10213},{0.102291},{-0.140604},{0.212434},{-0.0289296},{0.204658},{-0.220095},{-0.157856},{-0.226322},{0.0874405},{0.0473124},{0.0166551},{-0.228338},{0.0307165},{-0.0851658},{0.00148342}};
)";

	auto external = readCSV(argv[2]);
	coconut << "StaticVector<double," << external.size() << "> external{\n";
    for (auto feat : external) {
		coconut << feat[0] << ",";
	}
	coconut << "\n};\n";

	coconut << R"(
auto w2v_map = loadWord2Vec("../aquaint+wiki.txt.gz.ndim=50.bin");

auto convolve = [](auto input, auto filter) {
  HybridVector<double, MAX_SENTENCE_LENGTH> result(MAX_SENTENCE_LENGTH);
  for (int i = 0; i < input.columns() - COLUMN_PADDING; i++) {
    auto sub = submatrix(input, 0, i, filter.rows(), filter.columns());
    auto cc = sub % filter;
    double sum = 0;
    for (int j = 0; j < cc.rows(); j++) {
      for (auto it = cc.begin(j); it != cc.end(j); ++it) {
        sum += *it;
      }
    }
    result[i] = sum;
  }
  result.resize(input.columns());
  return result;
};

auto start = std::chrono::steady_clock::now();

)";

	for (auto part : {"question", "answer"}) {
		/* Load the query terms into a vector */
		coconut << "std::string " << part << "_line;\n";
		coconut << "getline(std::cin, " << part << "_line);\n";
		coconut << "std::stringstream " << part << "_ss(" << part << "_line);\n";
		coconut << "std::vector<std::string> " << part << "_words{std::istream_iterator<std::string>{" << part << "_ss}, std::istream_iterator<std::string>{}};\n";
		coconut << "\n";
        /* Create a matrix that's big enough for the query and padding */
		coconut << "HybridMatrix<float, EMBED_DIMENSION, MAX_SENTENCE_LENGTH + COLUMN_PADDING * 2> " << part << "(EMBED_DIMENSION, MAX_SENTENCE_LENGTH + COLUMN_PADDING * 2);\n";
        /* Set the relevant columns in the matrix to be the word2vec values, or if we can't find it, a random vector */
		coconut << "for (int i = 0; i < " << part << "_words.size(); i++) {\n";
		coconut << "  auto w2v_p = w2v_map.find(" << part << "_words[i]);\n";
		coconut << "  auto w2v = w2v_p == w2v_map.end() ? unknown_" << part << "_word : w2v_p->second;\n";
		coconut << "  submatrix(" << part << ", 0, i + COLUMN_PADDING, EMBED_DIMENSION, 1) = w2v;\n";
		coconut << "}\n";
        /* Reshape it to match the number of terms given */
		coconut << "" << part << ".resize(EMBED_DIMENSION, " << part << "_words.size() + 2 * COLUMN_PADDING);\n";
        coconut << "\n";
        /* Perform the convolutions */
		coconut << "StaticVector<double, " << part << "_convolution_biases.size()> " << part << "_conv_map;\n";
		coconut << "for (int i = 0; i < " << part << "_convolution_filters.size(); i++) {\n";
		coconut << "  " << part << "_conv_map[i] = max(convolve(" << part << ", " << part << "_convolution_filters[i]));\n";
		coconut << "}\n";
		coconut << part << "_conv_map = tanh(" << part << "_conv_map + " << part << "_convolution_biases);\n";
	}

	coconut << R"(
StaticVector<double, question_convolution_biases.size() + answer_convolution_biases.size() + external.size()> joinLayer;
subvector(joinLayer, 0, question_conv_map.size()) = question_conv_map;
subvector(joinLayer, question_conv_map.size(), answer_conv_map.size()) = answer_conv_map;
subvector(joinLayer, question_conv_map.size() + answer_conv_map.size(), external.size()) = external;

auto HiddenLayer = tanh((trans(joinLayer) * hidden_layer_weights) + trans(hidden_layer_biases)) * 2;
auto FinalLayer = trans((HiddenLayer * trans(softmax_layer_weights)) + trans(softmax_layer_biases));

StaticVector<double, 2> fmax(max(FinalLayer));
auto submax = FinalLayer - fmax;
auto expsubmax = exp(submax);
auto sumexpsubmax = expsubmax[0] + expsubmax[1];

auto end = std::chrono::steady_clock::now();
std::chrono::duration<double, std::milli> time = end - start;
std::cout << "Prepping matrices + forward: " << time.count() << "ms\n";

std::cout << "Final values: " << submax[0] - log(sumexpsubmax) << ", " << submax[1] - log(sumexpsubmax) << std::endl;
}
)";

    return EXIT_SUCCESS;
}
