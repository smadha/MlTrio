package edu.usc.mltrio.recommender;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Run user based CF on test data
 */
public class MahoutPredictUserBasedRecommender {

	public static void main(String[] args) throws Exception {
		DataModel model = new FileDataModel(
				new File("/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/invited_info_train_mahout.csv"));

		Recommender recommender = new MahoutUserBasedRecommender(1024, 0.01).buildRecommender(model);

		String csvFile = "/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/validate_nolabel_mahout.csv";
		String line = "";
		String cvsSplitBy = ",";

		BufferedReader br = new BufferedReader(new FileReader(csvFile));
		while ((line = br.readLine()) != null) {
			// use comma as separator
			String[] userAndQues = line.split(cvsSplitBy);
			long userID = Long.parseLong(userAndQues[0]);
			long quesID = Long.parseLong(userAndQues[1]);
			float pred;
			try {
				pred = recommender.estimatePreference(userID, quesID);
			} catch (Exception e) {
				pred = 0.1f;
			}
			
			if ( Float.isNaN(pred) ){
				pred = 0.1f;
			}
			
			DecimalFormat df = new DecimalFormat("#0.0000");
			
			System.out.println(userID + "," + quesID + "," + df.format(pred/4.1));
		}
		
		br.close();
	}
}
