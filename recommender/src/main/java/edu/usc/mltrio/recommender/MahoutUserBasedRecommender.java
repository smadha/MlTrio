package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * Find best num_neigh and similarity matrix using validation
 * Neigh - 1110 Acc - 0.6704911610004202
 * Neigh - 110 Acc - 0.6675216277313736
 * Neigh - 710 Acc - 0.6682084557050264
 * 
 * Neigh - 100 min_similarity - 2.401000000000001 Acc - 0.6735679866561057
 * Neigh - 100 min_similarity - 0.8009999999999999 Acc - 0.6751737932823109
 * Neigh - 700 min_similarity - 0.101 Acc - 0.670136541986574
 * Neigh - 700 min_similarity - 0.8009999999999999 Acc - 0.6814502761120604 	BEST
 * Neigh - 1100 min_similarity - 0.30100000000000005 Acc - 0.6721438656928943
 * Similarity TanimotoCoefficientSimilarity
 */
public class MahoutUserBasedRecommender implements RecommenderBuilder{
	int num_neigh = 1000;
	double min_similarity = 0.1;
	
	public MahoutUserBasedRecommender(int num_neigh, double min_similarity) {
		this.num_neigh = num_neigh;
		this.min_similarity = min_similarity;
	}
	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File("/Users/madhav/Documents/workspace/ml/MlTrio/bytecup2016data/invited_info_train_mahout.csv"));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		int [] num_neigh_arr = {700,1100};
		
		for(int num_neigh : num_neigh_arr){
			for(double min_similarity = 0.001; min_similarity<1.5; min_similarity += 0.1 ){
				RecommenderBuilder builder = new MahoutUserBasedRecommender(num_neigh, min_similarity);
				
				double result = evaluator.evaluate(builder, null, model, 0.8, 0.8);
				System.out.println("Neigh - "+ num_neigh + " min_similarity - "+ min_similarity + " Acc - " + result);
			}
		}

	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

//		TanimotoCoefficientSimilarity UncenteredCosineSimilarity  LogLikelihoodSimilarity
		UserSimilarity similarity = new TanimotoCoefficientSimilarity(model);
		
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(num_neigh, similarity, model);
//		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.3, similarity, model);
		
		return new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity);
	}
}
