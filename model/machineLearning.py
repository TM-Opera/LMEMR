import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------ Machine Learning Baseline Models ------------------

# ================================
# Random Forest Regressor Model
# ================================
class RandomForestRegressorModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X, y_reg):
        """
        Train the Random Forest regression model.

        Args:
            X: [B, 6167] Concatenated 2D feature matrix
            y_reg: [B, 24] Continuous target values (24 time steps per sample)
        """
        B, D = X.shape
        assert D == 6167, f"Expected X dim 6167, got {D}"

        # Repeat X 24 times to match the number of time steps
        X_expanded = np.repeat(X, 24, axis=0)  # [B*24, 6167]
        
        # Flatten y_reg into [B*24]
        y_flat = y_reg.reshape(-1)  # [B*24]

        # Standardize X
        X_scaled = self.scaler.fit_transform(X_expanded)  # [B*24, 6167]

        # Train the model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y_flat)
        print("✅ Random Forest model training completed")

    def predict(self, X):
        """
        Predict regression values.

        Args:
            X: [B, 6167] Input features

        Returns:
            pred: [B, 24] Predicted values for each of the 24 time steps
        """
        B, D = X.shape
        assert D == 6167, f"Expected X dim 6167, got {D}"

        # Repeat X 24 times to match the number of time steps
        X_expanded = np.repeat(X, 24, axis=0)  # [B*24, 6167]

        # Standardize (using scaler fitted on training data)
        X_scaled = self.scaler.transform(X_expanded)

        # Predict [B*24]
        pred_flat = self.model.predict(X_scaled)  # [B*24]

        # Reshape to [B, 24]
        return pred_flat.reshape(B, 24)


# ================================
# Linear Regression Model
# ================================
class LinearRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X, y_reg):
        """
        Train the Linear Regression model.

        Args:
            X: [B, 6167] Concatenated 2D feature matrix
            y_reg: [B, 24] Continuous target values
        """
        B, D = X.shape
        assert D == 6167, f"Expected X dim 6167, got {D}"

        # Repeat X 24 times to match the number of time steps
        X_expanded = np.repeat(X, 24, axis=0)  # [B*24, 6167]
        
        # Flatten labels
        y_flat = y_reg.reshape(-1)  # [B*24]

        # Standardize
        X_scaled = self.scaler.fit_transform(X_expanded)  # [B*24, 6167]

        # Train the model
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X_scaled, y_flat)
        print("✅ Linear Regression model training completed")

    def predict(self, X):
        """
        Predict regression values.

        Args:
            X: [B, 6167] Input features

        Returns:
            pred: [B, 24] Predicted values for each of the 24 time steps
        """
        B, D = X.shape
        assert D == 6167, f"Expected X dim 6167, got {D}"

        # Repeat X 24 times to match the number of time steps
        X_expanded = np.repeat(X, 24, axis=0)  # [B*24, 6167]

        # Standardize
        X_scaled = self.scaler.transform(X_expanded)

        # Predict and reshape
        pred_flat = self.model.predict(X_scaled)  # [B*24]
        return pred_flat.reshape(B, 24)
    

# ================================
# Modified evaluate_all_metrics: returns pred_classes and true_cls for plotting
# ================================

def evaluate_all_metrics(
    y_pred_reg,
    y_true_reg,
    y_true_class,
    bin_centers,
    processor=None
):
    """
    Compute unified metrics: R², MAE, MAPE, OA, and return data for confusion matrix.

    Returns:
        dict: {'r2': float, 'mae': float, 'mape': float, 'oa': float, 
               'pred_classes': np.array, 'true_classes': np.array}
    """
    def to_numpy(x):
        if isinstance(x, list):
            x = np.array(x)
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        return x

    y_pred_reg = to_numpy(y_pred_reg)
    y_true_reg = to_numpy(y_true_reg)
    y_true_class = to_numpy(y_true_class)

    pred_flat = y_pred_reg.flatten()      # [B*T]
    true_reg_flat = y_true_reg.flatten()  # [B*T]
    true_cls_flat = y_true_class.flatten()  # [B*T]

    if not (len(pred_flat) == len(true_reg_flat) == len(true_cls_flat)):
        raise ValueError(f"Length mismatch: "
                         f"pred={len(pred_flat)}, "
                         f"true_reg={len(true_reg_flat)}, "
                         f"true_cls={len(true_cls_flat)}")

    mask = ~(np.isnan(pred_flat) | np.isinf(pred_flat) |
             np.isnan(true_reg_flat) | np.isinf(true_reg_flat) |
             np.isnan(true_cls_flat))
    
    if not np.any(mask):
        return {
            'r2': np.nan, 'mae': np.nan, 'mape': np.nan, 'oa': np.nan,
            'pred_classes': None, 'true_classes': None
        }

    y_pred = pred_flat[mask]
    y_true_reg = true_reg_flat[mask]
    y_true_cls = true_cls_flat[mask].astype(int)

    # R² Score
    try:
        r2 = r2_score(y_true_reg, y_pred)
    except:
        r2 = np.nan

    # MAE
    mae = mean_absolute_error(y_true_reg, y_pred)

    # MAPE
    y_true_filtered = processor.inverse_transform(y_true_reg)
    y_pred_filtered = processor.inverse_transform(y_pred)
    mask_zero = y_true_filtered > 0
    if not np.any(mask_zero):
        mape = np.nan
    else:
        mape = np.mean(np.abs(y_pred_filtered - y_true_filtered) / y_true_filtered) * 100

    # OA & class predictions for confusion matrix
    centers = np.array(bin_centers)
    C = len(centers)
    diff = np.abs(y_pred[:, None] - centers[None, :])  # [N, C]
    pred_classes = np.argmin(diff, axis=1).astype(int)  # [N]

    valid_cls_mask = (pred_classes >= 0) & (pred_classes < C) & \
                     (y_true_cls >= 0) & (y_true_cls < C)
    if not np.any(valid_cls_mask):
        oa = np.nan
        pred_classes_final = true_cls_final = None
    else:
        oa = (pred_classes[valid_cls_mask] == y_true_cls[valid_cls_mask]).mean()
        pred_classes_final = pred_classes[valid_cls_mask]
        true_cls_final = y_true_cls[valid_cls_mask]

    return {
        'r2': round(r2, 4) if not np.isnan(r2) else np.nan,
        'mae': round(mae, 4),
        'mape': round(mape, 2) if not np.isnan(mape) else np.nan,
        'oa': round(oa, 4) if not np.isnan(oa) else np.nan,
        'pred_classes': pred_classes_final,
        'true_classes': true_cls_final
    }


def train_and_evaluate_ml_model(config):
    """
    Train and evaluate the model, while plotting the confusion matrix on the validation set.
    """
    seed = config.get('seed', 69)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = config.get('model')
    full_loader = config.get('dataloader')
    bin_centers = np.array(config.get('bin_centers'))
    processor = config.get('processor', None)

    if model is None:
        raise ValueError("❌ config must contain 'model'")
    if full_loader is None:
        raise ValueError("❌ config must contain 'dataloader'")
    if processor is None:
        raise ValueError("❌ config must contain 'processor'")

    print("🚀 Loading data from DataLoader...")
    data_batch = next(iter(full_loader))

    try:
        rs_feat = data_batch['img_code']['rs_code']
        svi_feat = data_batch['svi']
        building_feat = data_batch['img_code']['building_code']
        poi_feat = data_batch['pois']
    except KeyError as e:
        raise KeyError(f"Missing key feature field: {str(e)}")

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)

    rs_feat = to_numpy(rs_feat)
    svi_feat = to_numpy(svi_feat)
    building_feat = to_numpy(building_feat)
    poi_feat = to_numpy(poi_feat)

    N = rs_feat.shape[0]
    if svi_feat.ndim == 3:
        svi_feat = np.mean(svi_feat, axis=1)

    assert all(x.shape[0] == N for x in [rs_feat, svi_feat, building_feat, poi_feat])
    assert svi_feat.shape == (N, 2048), f"svi_feat shape error: {svi_feat.shape}"

    print(f"📌 Number of nodes: {N}")
    print(f"✅ Four-modal static feature extraction completed:")
    print(f"   - RS Code:       {rs_feat.shape}")
    print(f"   - SVI:           {svi_feat.shape}")
    print(f"   - Building Code: {building_feat.shape}")
    print(f"   - POIs:          {poi_feat.shape}")

    X_full = np.concatenate([rs_feat, svi_feat, building_feat, poi_feat], axis=1)
    print(f"✅ Feature concatenation completed: X_full.shape = {X_full.shape}")

    try:
        y_reg_tensor = data_batch['targets']['normalized'].squeeze()
        y_class_tensor = data_batch['targets']['labels'].squeeze()
    except KeyError as e:
        raise KeyError(f"Missing target label field: {str(e)}")

    y_reg = to_numpy(y_reg_tensor)
    y_class = to_numpy(y_class_tensor)

    print(f"✅ Target labels extracted: y_reg.shape = {y_reg.shape}, y_class.shape = {y_class.shape}")

    train_ratio = config.get('train_ratio', 0.6)
    val_ratio = config.get('val_ratio', 0.2)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("Train + validation ratio cannot exceed 1.0")

    perm = np.random.permutation(N)
    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)

    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True

    print(f"✅ Data split complete: Training {train_mask.sum()} | Validation {val_mask.sum()}")

    X_train_nodes = X_full[train_mask]
    y_reg_train_2d = y_reg[train_mask]
    y_class_train_2d = y_class[train_mask]

    X_val_nodes = X_full[val_mask]
    y_reg_val_2d = y_reg[val_mask]
    y_class_val_2d = y_class[val_mask]

    print(f"✅ Training set shape: X_train: {X_train_nodes.shape}, y_reg_train: {y_reg_train_2d.shape}")
    print(f"✅ Validation set shape: X_val: {X_val_nodes.shape}, y_reg_val: {y_reg_val_2d.shape}")

    if len(X_train_nodes) == 0:
        raise ValueError("❌ Training set is empty, please check train_ratio")
    if len(X_val_nodes) == 0:
        raise ValueError("❌ Validation set is empty, please check val_ratio")

    print("🧠 Starting model training...")
    try:
        model.train(X_train_nodes, y_reg=y_reg_train_2d)
        print("✅ Custom model training completed")
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'r2': np.nan, 'mae': np.nan, 'mape': np.nan, 'oa': np.nan}

    print("🔍 Making predictions on validation set...")
    try:
        y_pred_2d = model.predict(X_val_nodes)
        y_pred_flat = y_pred_2d.reshape(-1)
    except Exception as e:
        print(f"❌ Model prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'r2': np.nan, 'mae': np.nan, 'mape': np.nan, 'oa': np.nan}

    y_reg_val_flat = y_reg_val_2d.reshape(-1)
    y_class_val_flat = y_class_val_2d.reshape(-1)

    metrics = evaluate_all_metrics(
        y_pred_reg=y_pred_flat,
        y_true_reg=y_reg_val_flat,
        y_true_class=y_class_val_flat,
        bin_centers=bin_centers,
        processor=processor
    )

    print("\n" + "=" * 50)
    print("📊 Validation Set Performance Metrics")
    print("=" * 50)
    print(f"  R² Score:          {metrics['r2']:6.4f}")
    print(f"  MAE:               {metrics['mae']:6.4f}")
    print(f"  MAPE:              {metrics['mape']:6.2f}%")
    print(f"  Overall Accuracy:  {metrics['oa']:6.4f}")
    print("=" * 50)

    # ================================
    # 📊 Plot Confusion Matrix (only if valid class data exists)
    # ================================
    if metrics['pred_classes'] is not None and metrics['true_classes'] is not None:
        cm = confusion_matrix(metrics['true_classes'], metrics['pred_classes'], labels=list(range(10)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(10)), yticklabels=list(range(10)))
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix on Validation Set')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Cannot plot confusion matrix: no valid class data available")

    return metrics