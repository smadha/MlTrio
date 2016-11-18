package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Find best num_neigh and similarity matrix using validation
 * 
 * numFeatures - 64 lambda - 1.0E-4 Acc - 0.3511877294688672
 * numFeatures - 64 lambda - 0.001 Acc - 0.19067660161192174
 * 
 */
public class MahoutMatFactRecommender implements RecommenderBuilder{
	
	int numFeatures = 50; 
	double lambda = 0.1; 
	int numIterations = 50;
	
	public MahoutMatFactRecommender(int numFeatures, double lambda, int numIterations) {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.numIterations = numIterations;
	}

	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		int [] numFeaturesArr = {16,32,64,128,512};
		double [] lambdaArr = {0.00001,0.0001, 0.001, 0.01};
		
		for(int numFeaturesI : numFeaturesArr){
			for(double lambdaI : lambdaArr ){
				RecommenderBuilder builder = new MahoutMatFactRecommender(numFeaturesI, lambdaI, 50);
				
				double result = evaluator.evaluate(builder, null, model, 0.8, 0.8);
				System.out.println("numFeatures - "+ numFeaturesI + " lambda - "+ lambdaI + " Acc - " + result);
			}
		}
	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

		ALSWRFactorizer factorizer = new ALSWRFactorizer(model, numFeatures, lambda, numIterations);

		return new SVDRecommender(model, factorizer);
	}
}
