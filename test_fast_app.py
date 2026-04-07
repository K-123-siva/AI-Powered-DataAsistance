"""
Unit Tests for AI Data Science Assistant
Tests core functionality, data processing, and model training
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import pickle
import io


class TestDataLoading:
    """Test data loading and validation"""
    
    def test_csv_loading(self):
        """Test CSV file loading"""
        # Create sample CSV
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Load CSV
        loaded_data = pd.read_csv(csv_buffer)
        
        assert loaded_data.shape == (5, 3)
        assert list(loaded_data.columns) == ['feature1', 'feature2', 'target']
        assert loaded_data['feature1'].sum() == 15
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame()
        assert df.empty
        assert df.shape == (0, 0)
    
    def test_missing_values_detection(self):
        """Test missing values detection"""
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, np.nan, 30, np.nan, 50]
        })
        
        missing_count = data.isnull().sum()
        assert missing_count['col1'] == 1
        assert missing_count['col2'] == 2
        assert data.isnull().sum().sum() == 3


class TestDataQuality:
    """Test data quality metrics"""
    
    def test_completeness_score(self):
        """Test data completeness calculation"""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, np.nan, 40, 50]
        })
        
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        assert completeness == 90.0
    
    def test_duplicate_detection(self):
        """Test duplicate row detection"""
        data = pd.DataFrame({
            'col1': [1, 2, 2, 3, 3],
            'col2': [10, 20, 20, 30, 30]
        })
        
        duplicate_count = data.duplicated().sum()
        assert duplicate_count == 2
    
    def test_uniqueness_score(self):
        """Test uniqueness score calculation"""
        data = pd.DataFrame({
            'col1': [1, 2, 2, 3, 4],
            'col2': [10, 20, 20, 30, 40]
        })
        
        duplicate_count = data.duplicated().sum()
        uniqueness = ((len(data) - duplicate_count) / len(data)) * 100
        
        assert uniqueness == 80.0


class TestStatisticalAnalysis:
    """Test statistical analysis functions"""
    
    def test_basic_statistics(self):
        """Test basic statistical calculations"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5]
        })
        
        assert data['values'].mean() == 3.0
        assert data['values'].median() == 3.0
        assert data['values'].std() == pytest.approx(1.5811, rel=1e-3)
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation"""
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        corr = data.corr()
        assert corr.loc['x', 'y'] == 1.0  # Perfect correlation
    
    def test_skewness_calculation(self):
        """Test skewness calculation"""
        # Right-skewed data
        data = pd.DataFrame({
            'values': [1, 1, 1, 2, 2, 3, 10]
        })
        
        skewness = data['values'].skew()
        assert skewness > 0  # Positive skew
    
    def test_outlier_detection_iqr(self):
        """Test IQR outlier detection"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
        
        Q1 = data['values'].quantile(0.25)
        Q3 = data['values'].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = data[(data['values'] < Q1 - 1.5*IQR) | 
                       (data['values'] > Q3 + 1.5*IQR)]
        
        assert len(outliers) >= 1
        assert 100 in outliers['values'].values


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, 
                                   n_informative=3, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        # Get importance
        importance = model.feature_importances_
        
        assert len(importance) == 5
        assert np.sum(importance) == pytest.approx(1.0, rel=1e-3)
        assert all(imp >= 0 for imp in importance)
    
    def test_correlation_threshold(self):
        """Test high correlation detection"""
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],  # Perfect correlation with x
            'z': [5, 4, 3, 2, 1]    # Negative correlation
        })
        
        corr_matrix = data.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append((corr_matrix.columns[i], 
                                    corr_matrix.columns[j], 
                                    corr_matrix.iloc[i, j]))
        
        assert len(high_corr) >= 1


class TestModelTraining:
    """Test machine learning model training"""
    
    def test_classification_model(self):
        """Test classification model training"""
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, 
                                   n_informative=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        assert accuracy > 0.5  # Better than random
        assert len(y_pred) == len(y_test)
    
    def test_regression_model(self):
        """Test regression model training"""
        # Create sample data
        X, y = make_regression(n_samples=100, n_features=5, 
                              noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        
        assert r2 > 0.5  # Reasonable fit
        assert len(y_pred) == len(y_test)
    
    def test_model_serialization(self):
        """Test model pickle serialization"""
        # Create and train model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Serialize
        model_bytes = pickle.dumps(model)
        
        # Deserialize
        loaded_model = pickle.loads(model_bytes)
        
        # Test predictions match
        pred_original = model.predict(X[:5])
        pred_loaded = loaded_model.predict(X[:5])
        
        assert np.array_equal(pred_original, pred_loaded)


class TestConfusionMatrix:
    """Test confusion matrix functionality"""
    
    def test_binary_confusion_matrix(self):
        """Test binary classification confusion matrix"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        
        tn, fp, fn, tp = cm.ravel()
        
        assert tn == 3  # True negatives
        assert fp == 1  # False positives
        assert fn == 1  # False negatives
        assert tp == 3  # True positives
    
    def test_multiclass_confusion_matrix(self):
        """Test multi-class confusion matrix"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3)
        assert np.trace(cm) >= 6  # At least 6 correct predictions
    
    def test_confusion_matrix_metrics(self):
        """Test metrics derived from confusion matrix"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestDataValidation:
    """Test data validation functions"""
    
    def test_numeric_column_detection(self):
        """Test numeric column detection"""
        data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        assert 'int_col' in numeric_cols
        assert 'float_col' in numeric_cols
        assert 'str_col' not in numeric_cols
    
    def test_categorical_column_detection(self):
        """Test categorical column detection"""
        data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'obj_col': ['x', 'y', 'z']
        })
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        assert 'str_col' in categorical_cols
        assert 'obj_col' in categorical_cols
        assert 'int_col' not in categorical_cols
    
    def test_data_type_validation(self):
        """Test data type validation"""
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        assert data['col1'].dtype in ['int64', 'int32']
        assert data['col2'].dtype == 'object'


class TestEncryption:
    """Test data encryption functionality"""
    
    def test_encryption_decryption(self):
        """Test encryption and decryption"""
        from cryptography.fernet import Fernet
        
        # Generate key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Original data
        original = "test data"
        
        # Encrypt
        encrypted = cipher.encrypt(original.encode())
        
        # Decrypt
        decrypted = cipher.decrypt(encrypted).decode()
        
        assert decrypted == original
        assert encrypted != original.encode()


class TestExportFunctionality:
    """Test export and download functionality"""
    
    def test_csv_export(self):
        """Test CSV export"""
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        csv_string = data.to_csv(index=False)
        
        assert 'col1,col2' in csv_string
        assert '1,a' in csv_string
    
    def test_model_export(self):
        """Test model export to pickle"""
        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Serialize
        model_bytes = pickle.dumps(model)
        
        assert len(model_bytes) > 0
        assert isinstance(model_bytes, bytes)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_row_dataframe(self):
        """Test handling of single row dataframe"""
        data = pd.DataFrame({
            'col1': [1],
            'col2': [2]
        })
        
        assert data.shape == (1, 2)
        assert data['col1'].mean() == 1
    
    def test_all_missing_column(self):
        """Test column with all missing values"""
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        assert data['col2'].isnull().all()
        assert data['col2'].isnull().sum() == 3
    
    def test_constant_column(self):
        """Test column with constant values"""
        data = pd.DataFrame({
            'col1': [5, 5, 5, 5, 5]
        })
        
        assert data['col1'].std() == 0
        assert data['col1'].nunique() == 1
    
    def test_high_cardinality_categorical(self):
        """Test high cardinality categorical column"""
        data = pd.DataFrame({
            'id': range(100)
        })
        
        unique_count = data['id'].nunique()
        cardinality = 'High' if unique_count > len(data) * 0.5 else 'Low'
        
        assert cardinality == 'High'


class TestPerformance:
    """Test performance with different dataset sizes"""
    
    def test_small_dataset(self):
        """Test with small dataset (< 100 rows)"""
        data = pd.DataFrame({
            'col1': range(50),
            'col2': range(50, 100)
        })
        
        assert data.shape[0] == 50
        corr = data.corr()
        assert corr is not None
    
    def test_medium_dataset(self):
        """Test with medium dataset (1000 rows)"""
        data = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000, 2000)
        })
        
        assert data.shape[0] == 1000
        stats = data.describe()
        assert stats is not None
    
    def test_many_columns(self):
        """Test with many columns"""
        data = pd.DataFrame(np.random.randn(100, 20))
        
        assert data.shape == (100, 20)
        corr = data.corr()
        assert corr.shape == (20, 20)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow"""
        # 1. Create data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        data['target'] = y
        
        # 2. Check data quality
        missing = data.isnull().sum().sum()
        assert missing == 0
        
        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop('target', axis=1), data['target'], 
            test_size=0.2, random_state=42
        )
        
        # 4. Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. Predict
        y_pred = model.predict(X_test)
        
        # 6. Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # 7. Export model
        model_bytes = pickle.dumps(model)
        
        assert accuracy > 0.5
        assert cm.shape == (2, 2)
        assert len(model_bytes) > 0
    
    def test_full_regression_workflow(self):
        """Test complete regression workflow"""
        # 1. Create data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        data['target'] = y
        
        # 2. Statistical analysis
        stats = data.describe()
        corr = data.corr()
        
        # 3. Train model
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop('target', axis=1), data['target'], 
            test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        assert r2 > 0.5
        assert stats is not None
        assert corr is not None


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
