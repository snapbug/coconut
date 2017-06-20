#include "avro/Compiler.hh"
#include "avro/DataFile.hh"
#include "avro/Decoder.hh"
#include "nnweights.hxx"
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
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

	std::vector<std::string> names{"question_convolution_filters", "question_convolution_biases",
	                               "answer_convolution_filters",   "answer_convolution_biases",
	                               "hidden_layer_weights",         "hidden_layer_biases",
	                               "softmax_layer_weights",        "softmax_layer_biases"};

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
			coconut << "std::vector<"
			        << "StaticMatrix<float, " << weight.dimension[1] << ", " << weight.dimension[2]
			        << ">> " << name << ";\n";
			coconut << name << ".reserve(" << weight.dimension[0] << ");\n";
			sz = weight.dimension[1] * weight.dimension[2];
			for (int n = 0; n < weight.dimension[0]; n++) {
				coconut << name << ".push_back(StaticMatrix<float, " << weight.dimension[1] << ", "
				        << weight.dimension[2] << ">{";
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
			coconut << "StaticMatrix<float, " << weight.dimension[0] << ", " << weight.dimension[1] << "> " << name << "{";
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
			coconut << "StaticVector<float, " << weight.dimension[0] << ", rowVector> " << name << "{";
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

	auto external = readCSV(argv[2]);
	coconut << "StaticVector<float," << external.size() << ", rowVector> external{\n";
	for (auto feat : external)
		coconut << feat[0] << ",";
	coconut << "\n};\n";

	/*
	 * Generate a "random" vector for unknown words -- while this _should_ be random I'm too dumb to
	 * work out how to use the same random number generator as numpy, so I borrowed this from the
	 * output of the weights for an example question...
	 */
	std::random_device r;
	std::mt19937 gen(r());
	gen.seed(1234);
	std::uniform_real_distribution<float> dist(-.25, .25);

	/*
     * This _should_ get the same unknown_word vector as the pytorch
     * version, but that's been a bit finnicky in the past, no gaurantees.
	 */
	coconut << "StaticMatrix<float, EMBED_DIMENSION, 1> unknown_word{";
	for (int i = 0; i < 50; i++)
		coconut << "{" << dist(gen) << "},";
	coconut << "};\n";

    coconut << "auto w2v_map = loadWord2Vec(\"../aquaint+wiki.txt.gz.ndim=50.bin\");\n";
    coconut << "while (std::cin) {\n";

	for (auto part : {"question", "answer"}) {
		/* Load the query terms into a vector */
		coconut << "std::string " << part << "_line;\n";
		coconut << "getline(std::cin, " << part << "_line);\n";
		coconut << "std::stringstream " << part << "_ss(" << part << "_line);\n";
		coconut << "std::vector<std::string> " << part << "_words{std::istream_iterator<std::string>{" << part << "_ss}, std::istream_iterator<std::string>{}};\n";
        coconut << "if (" << part << "_words.size() == 0) { break; }\n";
        /* Get vectors for the results from the convolutions */
		coconut << "StaticVector<float, " << part << "_convolution_biases.size(), rowVector> " << part << "_conv_map;\n";
        /* Create a matrix that's big enough for the query and padding */
        coconut << "HybridMatrix<float, EMBED_DIMENSION, MAX_SENTENCE_LENGTH + COLUMN_PADDING * 2> " << part << "_input(EMBED_DIMENSION, MAX_SENTENCE_LENGTH + COLUMN_PADDING * 2);\n";
        /* Left pad */
        coconut << "submatrix(" << part << "_input, 0, 0, EMBED_DIMENSION, COLUMN_PADDING) = 0;\n";
	}
    /* To store the convolution result */
    coconut << "HybridVector<float, MAX_SENTENCE_LENGTH, rowVector> conv_result(MAX_SENTENCE_LENGTH);\n";

	coconut << "auto start = std::chrono::steady_clock::now();\n";

    /* Prepare the input matrices for forwarding */
	for (auto part : {"question", "answer"}) {
		coconut << "\n";
        coconut << part << "_input.resize(EMBED_DIMENSION, MAX_SENTENCE_LENGTH + COLUMN_PADDING * 2);\n";
		/* Set the relevant columns in the matrix to be the word2vec values, or if we can't find it, a random vector */
		coconut << "for (int i = 0; i < " << part << "_words.size(); i++) {\n";
		coconut << "  auto w2v_p = w2v_map.find(" << part << "_words[i]);\n";
		coconut << "  auto w2v = w2v_p == w2v_map.end() ? unknown_word : w2v_p->second;\n";
		coconut << "  submatrix(" << part << "_input, 0, i + COLUMN_PADDING, EMBED_DIMENSION, 1) = w2v;\n";
		coconut << "}\n";
        /* Right pad */
        coconut << "submatrix(" << part << "_input, 0, " << part << "_words.size() + COLUMN_PADDING, EMBED_DIMENSION, COLUMN_PADDING) = 0;\n";
		/* Reshape it to match the number of terms given */
		coconut << "" << part << "_input.resize(EMBED_DIMENSION, " << part << "_words.size() + 2 * COLUMN_PADDING);\n";
    }

    /* Start the forwarding */
	for (auto part : {"question", "answer"}) {
		/* Perform the convolutions */
        coconut << "conv_result.resize(" << part << "_input.columns());\n";
		coconut << "for (int i = 0; i < " << part << "_convolution_filters.size(); i++) {\n";
        coconut << "  for (int k = 0; k < " << part << "_input.columns() - COLUMN_PADDING; k++) {\n";
        coconut << "    auto sub = submatrix(" << part << "_input, 0, k, " << part << "_convolution_filters[i].rows(), " << part << "_convolution_filters[i].columns());\n";
        coconut << "    auto cc = sub % " << part << "_convolution_filters[i];\n";
        coconut << "    float sum = 0.0;\n";
        coconut << "    for (int j = 0; j < cc.rows(); j++) {\n";
        coconut << "      sum += std::accumulate(cc.begin(j), cc.end(j), 0.0);\n";
        coconut << "    }\n";
        coconut << "  conv_result[k] = sum;\n";
        coconut << "  }\n";
        coconut << "  " << part << "_conv_map[i] = max(conv_result);\n";
		coconut << "}\n";
		coconut << part << "_conv_map = tanh(" << part << "_conv_map + " << part << "_convolution_biases);\n";
	}

	coconut << R"(
StaticVector<float, question_convolution_biases.size() + answer_convolution_biases.size() + external.size(), rowVector> joinLayer;
subvector(joinLayer, 0, question_conv_map.size()) = question_conv_map;
subvector(joinLayer, question_conv_map.size(), answer_conv_map.size()) = answer_conv_map;
subvector(joinLayer, question_conv_map.size() + answer_conv_map.size(), external.size()) = external;

auto HiddenLayer = tanh((joinLayer * hidden_layer_weights) + hidden_layer_biases) * 2;
auto FinalLayer = (HiddenLayer * trans(softmax_layer_weights)) + softmax_layer_biases;

StaticVector<float, 2, rowVector> fmax(max(FinalLayer));
auto submax = FinalLayer - fmax;
auto expsubmax = exp(submax);
auto sumexpsubmax = expsubmax[0] + expsubmax[1];

auto end = std::chrono::steady_clock::now();
std::chrono::duration<double, std::milli> time = end - start;

/* std::cout << time.count() << "\n"; */
/* std::cout << "Final values: " << submax[0] - log(sumexpsubmax) << ", " << submax[1] - log(sumexpsubmax) << std::endl; */
std::cout << submax[0] - log(sumexpsubmax) << " " << submax[1] - log(sumexpsubmax) << " " << time.count() << "\n";
}

}
)";

	return EXIT_SUCCESS;
}
