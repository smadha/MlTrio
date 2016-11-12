package edu.usc.mltrio.recommender;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Run user based CF on test data
 */
public class MahoutPredictUserBasedRecommender {

	private static final double DEFAULT_VALUE = 0.01;

	public static void main(String[] args) throws Exception {
		DataModel model = new FileDataModel(
				new File("/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/invited_info_train_mahout.csv"));

		Recommender recommender = new MahoutUserBasedRecommender(1024, DEFAULT_VALUE).buildRecommender(model);

		String csvFile = "/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/test_nolabel_mahout.csv";
		String line = "";
		String cvsSplitBy = ",";

		StringBuilder resBuilder = new StringBuilder("");

		double max = 0;

		List<Long> userIDList = new ArrayList<Long>();
		List<Long> quesIDList = new ArrayList<Long>();
		List<Double> predList = new ArrayList<Double>();

		BufferedReader br = new BufferedReader(new FileReader(csvFile));
		while ((line = br.readLine()) != null) {
			// use comma as separator
			String[] userAndQues = line.split(cvsSplitBy);
			long userID = Long.parseLong(userAndQues[0]);
			long quesID = Long.parseLong(userAndQues[1]);
			double pred;
			try {
				pred = recommender.estimatePreference(userID, quesID);
			} catch (Exception e) {
				pred = DEFAULT_VALUE;
			}

			if (Double.isNaN(pred)) {
				pred = DEFAULT_VALUE;
			}
			max = Math.max(max, pred);

			userIDList.add(userID);
			quesIDList.add(quesID);
			predList.add(pred);

		}

		br.close();
		
		System.out.println(max);
		for (int i = 0; i < userIDList.size(); i++) {

			Long userID = userIDList.get(i);
			Long quesID = quesIDList.get(i);
			Double pred = predList.get(i);

			pred = pred/max;
			// Done for making a classifier output
			if (pred >= 0.5) {
				pred = 1d;
			} else {
				pred = 0d;
			}

			DecimalFormat df = new DecimalFormat("#0.0000");

			resBuilder.append(userID + "," + quesID + "," + df.format(pred)).append("\n");
		}
		System.out.println(resBuilder.toString());
	}
}
