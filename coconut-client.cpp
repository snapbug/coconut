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

	while (std::cin) {
		std::string question;
		std::string answer;

		getline(std::cin, question);
		getline(std::cin, answer);

		if (question.size() > 0 && answer.size() > 0) {
			std::cout << "Question: '" << question << "'" << std::endl;
			std::cout << "Answer: '" << answer << "'" << std::endl;
			std::cout << "Score: " << qaClient.getScore(question, answer) << std::endl;
		}
	}
}
