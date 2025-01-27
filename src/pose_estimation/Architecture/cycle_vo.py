import torch
import torch.nn as nn



from .utils import *



class CycleVO(nn.Module):
    """Generator network with pose estimation capabilities."""
    def __init__(self, device, input_shape=(6, 256, 256), num_residual_block=9, condition_dim=7):
        """
        Args:
            device: Device to run the model on
            input_shape: Input tensor shape (channels, height, width)
            num_residual_block: Number of residual blocks in the generator
            condition_dim: Dimension of the condition vector (pose information)
        """
        super(CycleVO, self).__init__()
        self.condition_dim = condition_dim
        channels = input_shape[0]
        self.device = device
        self.skip_linear = None

        self.reproject = nn.Conv2d(256 + condition_dim, 256, kernel_size=1, stride=1, padding=0)

        # Initial Convolution Block
        out_features = 64
        self.initial_model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        in_features = out_features

        # Downsampling
        downsample_layers = []
        for _ in range(2):
            out_features *= 2
            downsample_layers += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        self.downsampling = nn.Sequential(*downsample_layers)

        # Pose Estimation Tail

        self.pose_conv = nn.Sequential(
            nn.Conv2d(in_features, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Dense part
        self.pose_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7)  # 3 for translation, 4 for rotation (quaternion)
        )

        # Residual blocks
        residual_layers = []
        for _ in range(num_residual_block):
            residual_layers += [ResidualBlock(out_features, out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2  # adjust for upsampling
            residual_layers += [
                nn.Upsample(scale_factor=2),  # --> width*2, height*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output Layer
        residual_layers += [nn.ReflectionPad2d(3),
                            nn.Conv2d(out_features, 3, 7),
                            nn.Tanh()
                            ]

        # Unpacking
        self.residual_and_upsampling = nn.Sequential(*residual_layers)

    def forward(self, x, c = None, mode="generate"):
        x = self.initial_model(x)
        x = self.downsampling(x)

        if mode == "pose":
            # pose = self.pose_estimation(x)
            conv_out = self.pose_conv(x)
            flattened = conv_out.view(conv_out.size(0), -1)

            # Skip connection
            concatenated = torch.cat([flattened, x.view(x.size(0), -1)], dim=1)  # concatenate the original input x with flattened output from conv

            # Initialize the linear layer if it hasn't been initialized
            if self.skip_linear is None:
                self.skip_linear = nn.Linear(concatenated.size(1), 7).to(self.device)

            skip_connection = self.skip_linear(concatenated)

            dense_out = self.pose_dense(conv_out)

            # Merge skip connection with the output of the dense layers
            pose = dense_out + skip_connection

            translation_part = pose[:, :3]
            rotation_part = normalize_quaternion(pose[:, 3:])
            rotation_matrix = quaternion_to_matrix(rotation_part)  # Assuming quaternion_to_matrix is defined elsewhere

            motion_matrix_SE3 = torch.eye(4).unsqueeze(0).repeat(rotation_matrix.shape[0], 1, 1)
            motion_matrix_SE3[:, :3, :3] = rotation_matrix
            motion_matrix_SE3[:, :3, 3] = translation_part

            return motion_matrix_SE3.to(self.device)

        elif mode == "generate":
            # Injecting condition after downsampling (bottleneck)
            c = motion_matrix_to_pose7(c).view(c.size(0), self.condition_dim, 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
            x = self.reproject(x)
            x = self.residual_and_upsampling(x)
            return x

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose between 'generate' and 'pose'.")


if __name__ == "__main__":
    # Test the models
    device = 'cpu'
    batch_size = 12
    image_size = 128

    # Create test models
    generator = CycleVO(input_shape=(6, image_size, image_size), device=device)

    # Create test input
    test_images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    stacked_images = torch.cat([test_images, test_images], dim=1)

    # Test forward passes
    gen_pose = generator(stacked_images, mode="pose")
    gen_img = generator(stacked_images, gen_pose, mode="generate")

    # Print output shapes
    print(f"Generated image shape: {gen_img.shape}")
    print(f"Generated pose shape: {gen_pose.shape}")