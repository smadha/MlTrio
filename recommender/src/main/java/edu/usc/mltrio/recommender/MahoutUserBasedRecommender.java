package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


public class MahoutUserBasedRecommender implements RecommenderBuilder{
	
	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File("/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/invited_info_train_mahout.csv"));
		
		Recommender recommender = new MahoutUserBasedRecommender().buildRecommender(model);

		System.out.println(recommender.estimatePreference(30,4739) );
		System.out.println(recommender.recommend(30,10) );
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		RecommenderBuilder builder = new MahoutUserBasedRecommender();
		double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
		System.out.println(result);


	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

//		TanimotoCoefficientSimilarity UncenteredCosineSimilarity
		UserSimilarity similarity = new UncenteredCosineSimilarity(model);
		
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(1000, similarity, model);
//		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.3, similarity, model);
		
		return new GenericUserBasedRecommender(model, neighborhood, similarity);
	}
}
