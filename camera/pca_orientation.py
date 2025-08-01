import numpy as np
from sklearn.decomposition import PCA

class PCACalculator:
    def __init__(self, n_components=3):
        self.pca = PCA(n_components=n_components)

    def perform_pca_analysis(self, data):
        self.pca.fit(data)
        transformed_data = self.pca.transform(data)
        principal_components = self.pca.components_
        orientation_angles = [np.arctan2(component[1], component[0]) for component in principal_components]
        return transformed_data, principal_components, orientation_angles

if __name__ == "__main__":
    masks = np.array([[333, 292], [332, 293], [332, 303], [335, 306], [337, 306], [338, 307], [367, 307], [368, 308], [373, 308], [374, 309], [376, 309], [377, 310], [386, 310], [387, 311], [395, 311], [396, 312], [401, 312], [402, 311], [404, 311], [405, 310], [406, 310], [406, 309], [403, 306], [403, 305], [402, 304], [402, 302], [401, 301], [401, 299], [400, 298], [400, 297], [399, 296], [399, 295], [397, 293], [396, 293], [395, 292]], dtype=np.float32)
    depth = np.array([0.5660000443458557, 0.5670000314712524, 0.5700000524520874, 0.5690000057220459, 0.5680000185966492, 0.5670000314712524, 0.5649999976158142, 0.5640000104904175, 0.5640000104904175, 0.5630000233650208, 0.5630000233650208, 0.562000036239624, 0.5610000491142273, 0.5600000023841858, 0.5580000281333923, 0.5560000538825989, 0.5550000071525574, 0.5560000538825989, 0.5570000410079956, 0.5590000152587891, 0.5590000152587891, 0.5600000023841858, 0.562000036239624, 0.562000036239624, 0.562000036239624, 0.5610000491142273, 0.5610000491142273, 0.562000036239624, 0.5630000233650208, 0.5670000314712524, 0.5700000524520874, 0.5730000138282776, 0.5820000171661377, 0.5830000042915344, 0.5910000205039978], dtype=np.float32)

    # Combine masks with depth to form the 3D coordinates
    boundary_points_3d = np.hstack((masks, depth.reshape(-1, 1)))

    pca_calculator = PCACalculator(n_components=3)
    transformed_data, principal_components, orientation_angles = pca_calculator.perform_pca_analysis(boundary_points_3d)

    print("Transformed Data:", transformed_data)
    print("Principal Components:", principal_components)
    print("Orientation Angles (in radians) in 3D:", orientation_angles)
