#include <iostream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/QuestionAnswering.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

int main(void) {
	boost::shared_ptr<TTransport> socket(new TSocket("localhost", 9090));
	boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
	boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

	QuestionAnsweringClient qaClient(protocol);

	transport->open();

	std::vector<std::string> answers{};
	std::string last_question{};
	std::vector<double> results;
	int pairs = 0;

	auto start = std::chrono::steady_clock::now();
	while (std::cin) {
		std::string question;
		std::string answer;

		getline(std::cin, question);
		getline(std::cin, answer);

#define BATCH_ANSWERS
#ifdef BATCH_ANSWERS
		if (question != last_question) {
			if (answers.size() > 0) {
				qaClient.getScores(results, last_question, answers);
				pairs += answers.size();
			}
			last_question = question;
			answers.clear();
		}
		answers.push_back(answer);
#else
        qaClient.getScore(question, answer);
        pairs++;
#endif
	}

    /*
     * Normally there would be another call to qaClient.getScores() here, but because the getline gets blank lines, we don't bother
     * this is testable by `assert(last_question == "")` not throwing.
     */
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> time = end - start;

	std::cout << pairs << " qa pairs in " << time.count() << "ms, or " << (1000 * (pairs / time.count())) << "qps\n";
}
