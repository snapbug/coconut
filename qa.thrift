namespace py qa

service QuestionAnswering {
    double getScore(1:string question, 2:string answer)
    list<double> getScores(1:string question, 2:list<string> answers)
}
