# This is the repo of Outlier Detection for Streaming Task Assignment in Crowdsourcing

In the whole, this repo can be divided into two parts. The first part is outlier detection. And the second part is task assignment.

To run these codes, only thing you can do is run the bash script "run.sh" in the top-level directory.

[1] In 3-5 lines, these codes are used to run the outlier detection algorithm, i.e., SA-Learn, which is proposed by our paper.
[2] In 11-12 lines, the anomaly scores returned by the SA-Learn is used to produce the anomaly-based worker-task weights. To excute these codes, it will be cost more than half hour.
[3] In 13-16 lines, four worker-task assignment algorithms, i.e., KM, DR, KM+outlier, DR+outlier, are adopted to do assignment.

As you can see in the "run.sh", [1] and [2] are commented. You can make a fresh start by uncommenting these code to do the outlier detection and preprocessing of worker-task assignment.
