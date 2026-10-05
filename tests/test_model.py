import unittest

import torch

from geneflowv2.model import (
    ExpressionRoutedGeneTokenEncoderV2,
    GeneFlowV2Model,
)


class ExpressionRoutedGeneTokenEncoderV2Test(unittest.TestCase):
    def make_encoder(self, max_genes=3):
        return ExpressionRoutedGeneTokenEncoderV2(
            num_genes=6,
            token_dim=32,
            output_dim=64,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
            max_cross_attention_genes=max_genes,
            cross_attention_expression_threshold=0.0,
        )

    def test_all_observed_genes_pool_but_only_top_nonzero_genes_route(self):
        torch.manual_seed(0)
        encoder = self.make_encoder(max_genes=2).eval()
        expression = torch.tensor(
            [[0.0, 0.1, 4.0, 2.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        observed = torch.ones_like(expression)
        conditioning = encoder.encode_conditioning(
            expression, observed, return_attention=True
        )

        self.assertEqual(tuple(conditioning.spatial_tokens.shape), (2, 7, 32))
        self.assertEqual(tuple(conditioning.spatial_token_mask.shape), (2, 7))
        self.assertTrue(torch.equal(conditioning.active_gene_mask[0], torch.tensor(
            [False, False, True, False, False, True]
        )))
        self.assertEqual(int(conditioning.active_gene_mask[1].sum()), 0)
        self.assertTrue(conditioning.spatial_token_mask[:, 0].all())
        self.assertTrue((conditioning.pooling_attention > 0).all())
        self.assertTrue(torch.allclose(
            conditioning.pooling_attention.sum(dim=1), torch.ones(2), atol=1e-5
        ))

    def test_missing_genes_do_not_pool_or_route_and_null_has_only_cell_token(self):
        torch.manual_seed(1)
        encoder = self.make_encoder(max_genes=0).eval()
        expression = torch.ones(2, 6)
        observed = torch.ones_like(expression)
        observed[0, -2:] = 0
        present = torch.tensor([True, False])
        conditioning = encoder.encode_conditioning(
            expression, observed, present, return_attention=True
        )

        self.assertTrue(torch.equal(
            conditioning.pooling_attention[0, -2:], torch.zeros(2)
        ))
        self.assertTrue(torch.equal(
            conditioning.active_gene_mask[0, -2:], torch.zeros(2, dtype=torch.bool)
        ))
        self.assertEqual(int(conditioning.spatial_token_mask[1].sum()), 1)
        self.assertEqual(int(conditioning.active_gene_mask[1].sum()), 0)

    def test_soft_routing_is_continuous_monotone_and_has_no_top_k_boundary(self):
        encoder = self.make_encoder(max_genes=2).eval()
        encoder.set_routing_mode("soft", soft_routing_temperature=1.0)
        observed = torch.ones(1, 6)
        low = torch.tensor([[0.0, 0.01, 0.1, 1.0, 2.0, 4.0]])
        high = low + torch.tensor([[0.0, 0.01, 0.1, 0.5, 0.5, 0.5]])
        low_conditioning = encoder.encode_conditioning(low, observed)
        high_conditioning = encoder.encode_conditioning(high, observed)
        low_weights = low_conditioning.spatial_token_mask[:, 1:]
        high_weights = high_conditioning.spatial_token_mask[:, 1:]

        self.assertEqual(low_weights.dtype, low.dtype)
        self.assertEqual(float(low_weights[0, 0]), 0.0)
        self.assertTrue(torch.all(high_weights >= low_weights))
        self.assertEqual(int((low_weights > 0).sum()), 5)
        self.assertGreater(int((low_weights > 0).sum()), encoder.max_cross_attention_genes)

        almost_same = encoder.encode_conditioning(low + 1e-5, observed)
        self.assertLess(
            float((almost_same.spatial_token_mask - low_conditioning.spatial_token_mask).abs().max()),
            2e-5,
        )

    def test_all_observed_routes_measured_zeros_but_not_missing_genes(self):
        torch.manual_seed(2)
        encoder = self.make_encoder(max_genes=2).eval()
        encoder.set_routing_mode("all_observed")
        expression = torch.tensor(
            [[0.0, 4.0, 0.0, 3.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        observed = torch.tensor(
            [[1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32
        )
        present = torch.tensor([True, False])

        conditioning = encoder.encode_conditioning(
            expression, observed, present, return_attention=True
        )

        expected = torch.tensor(
            [[True, True, True, False, True, False], [False] * 6]
        )
        self.assertTrue(torch.equal(conditioning.active_gene_mask, expected))
        self.assertEqual(conditioning.spatial_token_mask.dtype, torch.bool)
        self.assertTrue(conditioning.spatial_token_mask[:, 0].all())
        self.assertTrue(torch.equal(
            conditioning.spatial_token_mask[:, 1:], expected
        ))
        self.assertEqual(int(conditioning.active_gene_mask[0].sum()), 4)
        self.assertGreater(
            int(conditioning.active_gene_mask[0].sum()),
            encoder.max_cross_attention_genes,
        )

    def test_unknown_routing_mode_is_rejected(self):
        encoder = self.make_encoder()
        with self.assertRaisesRegex(ValueError, "all_observed"):
            encoder.set_routing_mode("unknown")


class GeneFlowV2ModelTest(unittest.TestCase):
    def make_model(self):
        return GeneFlowV2Model(
            rna_dim=8,
            img_channels=4,
            img_size=16,
            model_channels=32,
            num_res_blocks=1,
            attention_resolutions=(1,),
            dropout=0.0,
            channel_mult=(1, 1),
            num_heads=1,
            num_head_channels=32,
            gene_token_dim=32,
            gene_transformer_layers=1,
            gene_transformer_heads=4,
            gene_transformer_ff_dim=64,
            gene_transformer_dropout=0.0,
            max_cross_attention_genes=3,
            cross_attention_resolutions=(1, 2),
            cross_attention_heads=4,
            cross_attention_head_dim=8,
            cross_attention_dropout=0.0,
        )

    def test_forward_cfg_and_routed_attention_shapes(self):
        torch.manual_seed(2)
        model = self.make_model().eval()
        image = torch.randn(2, 4, 16, 16)
        time = torch.rand(2)
        expression = torch.tensor(
            [
                [0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0],
                [1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
            ]
        )

        with torch.no_grad():
            output, cell_attention, gene_attention, active = model(
                image,
                time,
                expression,
                return_cross_attention=True,
            )
            guided = model.forward_with_cfg(
                image, time, expression, guidance_scale=2.0
            )

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(guided.shape, image.shape)
        self.assertEqual(tuple(cell_attention.shape), (2,))
        self.assertEqual(tuple(gene_attention.shape), (2, 8))
        self.assertEqual(tuple(active.shape), (2, 8))
        self.assertTrue(torch.equal(gene_attention.masked_select(~active), torch.zeros(
            int((~active).sum())
        )))
        self.assertTrue(torch.allclose(
            cell_attention + gene_attention.sum(dim=1), torch.ones(2), atol=1e-5
        ))

    def test_conditioning_parameter_partition_excludes_image_backbone(self):
        model = self.make_model()
        condition_names = {
            name for name, _ in model.named_conditioning_parameters()
        }
        self.assertIn("rna_encoder.gene_identity", condition_names)
        self.assertTrue(any("cross_attention" in name for name in condition_names))
        self.assertTrue(any("rna_adapter" in name for name in condition_names))
        self.assertFalse(any(name.startswith("unet.input_blocks") for name in condition_names))
        self.assertFalse(any(name.startswith("unet.out.") for name in condition_names))
        self.assertLess(len(condition_names), len(list(model.named_parameters())))

    def test_soft_routing_forward_and_attention_probabilities(self):
        torch.manual_seed(3)
        model = self.make_model().eval()
        model.set_routing_mode("soft", soft_routing_temperature=1.5)
        image = torch.randn(2, 4, 16, 16)
        time = torch.rand(2)
        expression = torch.tensor(
            [
                [0.0, 0.1, 0.3, 0.0, 1.0, 0.0, 2.0, 4.0],
                [0.2, 0.0, 0.0, 0.4, 0.0, 1.0, 0.0, 3.0],
            ]
        )
        with torch.no_grad():
            output, cell_attention, gene_attention, active = model(
                image, time, expression, return_cross_attention=True
            )
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.equal(gene_attention.masked_select(~active), torch.zeros(
            int((~active).sum())
        )))
        self.assertTrue(torch.allclose(
            cell_attention + gene_attention.sum(dim=1), torch.ones(2), atol=1e-5
        ))

    def test_all_observed_mask_reaches_image_cross_attention(self):
        torch.manual_seed(5)
        model = self.make_model().eval()
        model.set_routing_mode("all_observed")
        image = torch.randn(2, 4, 16, 16)
        time = torch.rand(2)
        expression = torch.tensor(
            [
                [0.0, 2.0, 0.0, 5.0, 0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            ]
        )
        observed = torch.tensor(
            [
                [1, 1, 1, 0, 1, 0, 1, 1],
                [1, 0, 1, 1, 0, 1, 1, 1],
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            output, cell_attention, gene_attention, active = model(
                image,
                time,
                expression,
                gene_mask=observed,
                return_cross_attention=True,
            )

        expected = observed.bool()
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.equal(active, expected))
        self.assertTrue(torch.equal(
            gene_attention.masked_select(~expected),
            torch.zeros(int((~expected).sum())),
        ))
        self.assertTrue((gene_attention.masked_select(expected) > 0).all())
        self.assertTrue(torch.allclose(
            cell_attention + gene_attention.sum(dim=1), torch.ones(2), atol=1e-5
        ))

    def test_routing_mode_adds_no_checkpoint_parameters(self):
        hard = self.make_model()
        state = hard.state_dict()
        soft = self.make_model()
        soft.set_routing_mode("soft", soft_routing_temperature=1.0)
        soft.load_state_dict(state, strict=True)
        all_observed = self.make_model()
        all_observed.set_routing_mode("all_observed")
        all_observed.load_state_dict(state, strict=True)
        self.assertEqual(list(state), list(soft.state_dict()))
        self.assertEqual(list(state), list(all_observed.state_dict()))

    def test_ranking_autograd_can_be_restricted_to_conditioning_path(self):
        torch.manual_seed(4)
        model = self.make_model().train()
        image = torch.randn(2, 4, 16, 16)
        time = torch.rand(2) * 0.25
        expression = torch.rand(2, 8)
        target = torch.randn_like(image)


        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        warmup = (model(image, time, expression) - target).square().mean()
        warmup.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        correct = model(image, time, expression)
        shuffled = model(image, time, expression.flip(0))
        correct_mse = (correct - target).square().flatten(1).mean(1)
        shuffled_mse = (shuffled - target).square().flatten(1).mean(1)
        rank_loss = torch.nn.functional.softplus(
            (0.001 - (shuffled_mse - correct_mse)) / 0.001
        ).mean() * 0.001
        parameters = list(model.conditioning_parameters())
        gradients = torch.autograd.grad(rank_loss, parameters, allow_unused=True)
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                parameter.grad = gradient

        condition_names = {
            name for name, _ in model.named_conditioning_parameters()
        }
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
            for parameter in parameters
        ))
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if name not in condition_names
        ))


if __name__ == "__main__":
    unittest.main()
