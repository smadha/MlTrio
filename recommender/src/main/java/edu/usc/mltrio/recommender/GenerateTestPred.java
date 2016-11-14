package edu.usc.mltrio.recommender;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class GenerateTestPred {

	public static void main(String[] args) throws Exception{
		// LensKitRecommender
		Recommend rec  = new MahoutPredictMatFactRecommender();
		
		String csvFile = Constants.TEST_MAHOUT_CSV;
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
				pred = rec.predict(userID, quesID);
			} catch (Exception e) {
				pred = Constants.DEFAULT_VALUE;
			}

			if (Double.isNaN(pred)) {
				pred = Constants.DEFAULT_VALUE;
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
//			if (pred >= 0.5) {
//				pred = 1d;
//			} else {
//				pred = 0d;
//			}

			DecimalFormat df = new DecimalFormat("#0.0000");

			resBuilder.append(userID + "," + quesID + "," + df.format(pred)).append("\n");
		}
		System.out.println(resBuilder.toString());
	}
}
