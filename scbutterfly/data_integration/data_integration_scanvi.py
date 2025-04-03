import anndata
import scanpy as sc
import scvi
import os

#we want to integrathe the predicted data into the real ATAC. for this, we have merged the two datasets together just using anndata.concat functon (join = "outer"), since the two AnnData objects have the same dimensions 
#SCANVI integration can be used since we have cell type annotation in our data 
#this script will train the SCVI and then a SCANVI model and draw a UMAP based on the latent representation learnt by SCANVI

merged_atac = anndata.read_h5ad('./results/merged_atac_normal.h5ad')

merged_atac.layers["counts"] = merged_atac.X.copy()

sc.pp.highly_variable_genes(
    merged_atac,
    flavor="seurat_v3",
    n_top_genes=20000,
    layer="counts",
    batch_key="batch",
    subset=True
)

scvi_path = "./scvi_model2/"
scanvi_path = "./scanvi_model2/"

#load or train SCVI model
if os.path.exists(scvi_path):
    print("Loading pre-trained SCVI model...")
    scvi_model = scvi.model.SCVI.load(scvi_path, adata=merged_atac)
else:
    print("Training new SCVI model...")
    scvi.model.SCVI.setup_anndata(merged_atac, layer="counts", batch_key="batch", labels_key="cell_type")
    scvi_model = scvi.model.SCVI(merged_atac, n_layers=2, n_latent=30, gene_likelihood="nb")
    scvi_model.train()
    scvi_model.save(scvi_path, overwrite=True)
    
SCVI_LATENT_KEY = "X_scVI"
merged_atac.obsm[SCVI_LATENT_KEY] = scvi_model.get_latent_representation()
sc.pp.neighbors(merged_atac, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(merged_atac)
sc.tl.umap(merged_atac, min_dist=0.3)
sc.pl.umap(
    merged_atac,
    color=["batch", "cell_type"],
    frameon=False,
    ncols=1,
    save='merged_atac_normal_integrated_scvi.pdf'
)

#load or train SCANVI model
if os.path.exists(scanvi_path):
    print("Loading pre-trained SCANVI model...")
    scanvi_model = scvi.model.SCANVI.load(scanvi_path, adata=merged_atac)
else:
    print("Training new SCANVI model...")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, unlabeled_category="unknown")
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)
    scanvi_model.save(scanvi_path, overwrite=True)

SCANVI_LATENT_KEY = "X_scANVI"
merged_atac.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation()

merged_atac.write('./results/merged_atac_scanvi.h5ad')

sc.pp.neighbors(merged_atac, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(merged_atac, min_dist=0.3)
sc.pl.umap(
    merged_atac,
    color=["batch", "cell_type"],
    frameon=False,
    ncols=1,
    save='merged_atac_normal_integrated_scanvi.pdf'
)
