"""Build manual.pdf for the HDAC8 Assistant web application."""
from pathlib import Path

from fpdf import FPDF


class ManualPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "HDAC8 Assistant - User Manual", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 0, 0)
        w = self.w - self.l_margin - self.r_margin
        self.multi_cell(w, 7, title)
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        w = self.w - self.l_margin - self.r_margin
        self.multi_cell(w, 5, text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        w = self.w - self.l_margin - self.r_margin
        self.multi_cell(w, 5, f"  -  {text}")


def build_manual(output_path: Path) -> None:
    pdf = ManualPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 15, 18)

    # --- Page 1: Title & Overview ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "Manual of HDAC8 Assistant\nopen-source software")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "HDAC8 Assistant is a web application for the generation and evaluation of "
        "chemical compounds as potential HDAC8 inhibitors. The software combines "
        "de novo molecule generation (SMILES-RNN with reinforcement learning), "
        "QSAR-based activity prediction, applicability-domain assessment, "
        "drug-likeness evaluation (Muegge rules), and structural alert screening "
        "(Brenk filters and PAINS).",
    )
    pdf.ln(3)
    pdf.section_title("Appearance of the main window")
    pdf.body(
        "The main page displays five capability panels: Molecule generation, "
        "Machine learning (QSAR), OECD guidelines (applicability domain), "
        "Muegge's rules, and Structural Alerts. A download button provides this manual."
    )

    pdf.section_title("Step 1. Draw molecule or select input molecular files")
    pdf.body(
        "If you want to draw the structure of a chemical compound, use the built-in "
        "chemical editor Ketcher (https://github.com/epam/ketcher). After creating "
        "the structure, use two buttons: (1) \"Reset\" - delete the structure to "
        "create a new one; (2) \"Apply\" - transfer the structure for further analysis "
        "or generation in Step 2."
    )
    pdf.body(
        "After clicking the \"Apply\" button, make sure that the structure has been "
        "created. If successful, its SMILES will be displayed under the chemical editor."
    )
    pdf.body(
        "If you choose SMILES, paste the SMILES representation of the desired chemical "
        "structure and press Ctrl+Enter. If the entered structure is correct, the "
        "application will generate a 2D image of the compound. DON'T FORGET TO CLICK "
        "THE \"APPLY\" BUTTON when using the Ketcher editor."
    )

    # --- Page 2: Input methods ---
    pdf.add_page()
    pdf.body(
        "If the entered structure is incorrect, the application reports an error. "
        "If Ketcher export fails aromatic bond assignment (kekulization), switch to "
        "the SMILES input method and paste the same structure as text."
    )
    pdf.body(
        "If you choose a file *.sdf or *.csv containing multiple chemical structures, "
        "click the \"Browse files\" button and upload the file. For *.csv files, "
        "the file must contain a column named \"SMILES\"."
    )
    pdf.body(
        "If incorrect structures are detected in an *.sdf or *.csv file, the "
        "corresponding information appears in the section "
        "\"CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION\"."
    )

    pdf.section_title("Step 2. Select analysis mode")
    pdf.body(
        "In Step 2, choose one of the following modes from the drop-down menu:"
    )
    pdf.bullet(
        "HDAC8 - predict HDAC8 inhibitor activity (pIC50) for the input structure(s)."
    )
    pdf.bullet(
        "Molecule generation (SMILES-RNN) - generate new candidate HDAC8 inhibitors "
        "based on a reference structure from Step 1."
    )
    pdf.bullet(
        "Muegge rules, PAINS, Brenk structural alerts, Substructural search - "
        "evaluate drug-likeness and detect undesirable substructures."
    )

    # --- Page 3: HDAC8 prediction ---
    pdf.add_page()
    pdf.section_title("Step 2a. HDAC8 activity prediction")
    pdf.body(
        "Select \"HDAC8\" and click the \"Run predictions!\" button. "
        "The QSAR model uses ECFP4 molecular fingerprints and a CatBoost gradient "
        "boosting regressor trained on curated HDAC8 activity data from scientific "
        "literature and ChEMBL."
    )
    pdf.section_title("Step 3. Prediction results (HDAC8 mode)")
    pdf.body(
        "The form of presentation depends on the input format. For a single molecule "
        "(Ketcher or SMILES), results are displayed on screen. For *.sdf or *.csv files, "
        "results for valid structures are shown in a downloadable table."
    )
    pdf.body("The final table \"Prediction results\" contains the following columns:")
    pdf.bullet("SMILES - chemical structure in SMILES notation.")
    pdf.bullet(
        "Predicted value pIC50 - predicted HDAC8 inhibitory activity (negative "
        "logarithm of IC50 in molar concentration). If experimental data is available "
        "in ChEMBL, the label \"see experimental value\" is displayed."
    )
    pdf.bullet(
        "Applicability domain_HDAC8 - compliance with the model applicability domain "
        "(AD). If experimental data is available in ChEMBL, \"-\" is displayed."
    )
    pdf.bullet(
        "Experimental value, pIC50 - experimental data from ChEMBL (average if multiple "
        "values exist)."
    )
    pdf.bullet(
        "STD - standard deviation of experimental activity values from ChEMBL."
    )
    pdf.bullet("chembl_ID - ChEMBL identifier of the molecule.")

    # --- Page 4: Molecule generation ---
    pdf.add_page()
    pdf.section_title("Step 2b. Molecule generation (SMILES-RNN)")
    pdf.body(
        "Select \"Molecule generation (SMILES-RNN)\" in Step 2. This mode generates "
        "new candidate HDAC8 inhibitors using a SMILES-RNN model pretrained on "
        "drug-like compounds from ChEMBL. The model is optimized via reinforcement "
        "learning (REINVENT algorithm) to maximize predicted pIC50 values while "
        "ensuring applicability-domain compliance. An ECFP4-based CatBoost QSAR model "
        "serves as the core of the RL reward function."
    )
    pdf.body("Input requirements:")
    pdf.bullet(
        "Single reference: draw a molecule in Ketcher or enter one SMILES in Step 1."
    )
    pdf.bullet(
        "Multiple references: upload a *.csv file with a \"SMILES\" column; each valid "
        "reference runs a separate RL job."
    )
    pdf.body("Generation parameters (adjustable before running):")
    pdf.bullet(
        "Maximum molecules to return (default 500) - per reference, the maximum number "
        "of AD-compliant candidates kept from each RL run."
    )
    pdf.bullet("RL n_steps (default 250) - number of reinforcement-learning iterations.")
    pdf.bullet("RL batch_size (default 128) - batch size for each RL step.")
    pdf.bullet(
        "Molecules in final merged table (CSV mode only, default 25) - after merging "
        "all references, keep the top N analogs ranked by predicted pIC50."
    )
    pdf.body(
        "Click \"Run generation!\" to start the pipeline. GPU is used automatically "
        "when available; otherwise CPU execution is selected. Generation may take "
        "several minutes depending on hardware and parameter settings."
    )

    # --- Page 5: Generation results & filtering ---
    pdf.add_page()
    pdf.section_title("Step 3. Generation results")
    pdf.body(
        "Generated structures undergo deduplication, validity checks, and optional "
        "ring-system filtering. Only molecules inside the QSAR applicability domain "
        "(in_AD >= 0.5) are retained. Compounds matching structures in the experimental "
        "HDAC8 dataset (HDAC8_exp_data_inchi.csv) are excluded. Results are ranked "
        "by predicted pIC50 (descending)."
    )
    pdf.body("The results table contains the following columns:")
    pdf.bullet("No. - row number.")
    pdf.bullet("generated_smiles - SMILES string produced by the generative model.")
    pdf.bullet("canonical_smiles - canonical SMILES after RDKit standardization.")
    pdf.bullet("predicted_pIC50 - QSAR-predicted HDAC8 inhibitory activity.")
    pdf.bullet(
        "SAScore - Synthetic Accessibility Score (Ertl & Schuffenhauer, 2009); "
        "lower values indicate easier synthetic access."
    )
    pdf.bullet("in_AD - applicability-domain flag (True if inside AD).")
    pdf.bullet(
        "Muegge rules - compliance with Muegge drug-likeness criteria "
        "(Compliant / Non-compliant with violation details)."
    )
    pdf.body(
        "Summary metrics are displayed above the table: initial generated count, "
        "valid unique structures, inside-AD count, and excluded experimental matches. "
        "An expandable \"Pipeline details\" section shows RL parameters and run "
        "directories. Click \"Download generated molecules as CSV\" to save results."
    )
    pdf.body(
        "For multi-reference CSV input, results from all references are merged, "
        "deduplicated by canonical SMILES, sorted by predicted pIC50, and truncated "
        "to the chosen top N rows."
    )

    # --- Page 6: Muegge, Brenk, PAINS ---
    pdf.add_page()
    pdf.section_title("Step 2c. Muegge rules and structural alerts")
    pdf.body(
        "Select \"Muegge rules, PAINS, Brenk structural alerts, Substructural search\" "
        "and click \"Run predictions!\". HDAC8 Assistant evaluates compliance with "
        "Muegge's drug-likeness rules. Results are displayed in tabular form and as a "
        "Bioavailability Radar for visual assessment of lipophilicity (log P), "
        "molecular weight (MW), hydrogen bond donors (HBD), hydrogen bond acceptors "
        "(HBA), rotatable bonds, TPSA, and ring count against Muegge limits."
    )
    pdf.body(
        "The application also detects widely used medicinal-chemistry structural alerts:"
    )
    pdf.bullet(
        "Brenk filters - 105 fragments identified by Brenk et al. [Brenk, R. et al. "
        "Lessons learnt from assembling screening libraries for drug discovery for "
        "neglected diseases. ChemMedChem 3, 435-444 (2008).] as putatively toxic, "
        "chemically reactive, metabolically unstable, or associated with poor "
        "pharmacokinetics."
    )
    pdf.bullet(
        "PAINS (pan-assay interference compounds) - substructures identified by "
        "Baell et al. [Baell, J. B. & Holloway, G. A. New substructure filters for "
        "removal of pan-assay interference compounds (PAINS) from screening libraries "
        "and for their exclusion in bioassays. J. Med. Chem. 53, 2719-2740 (2010).] "
        "that yield false-positive biological responses."
    )
    pdf.body(
        "HDAC8 Assistant returns warnings if such moieties are found in the molecule "
        "under evaluation."
    )
    pdf.ln(4)
    pdf.section_title("Software requirements for molecule generation")
    pdf.body(
        "Molecule generation requires the Python packages smiles-rnn, molscore, and "
        "PyTorch (see requirements.txt). Optional packages useful-rdkit-utils (ring-system "
        "filtering) and sascorer (SAScore calculation) enhance post-processing when "
        "available."
    )

    pdf.output(str(output_path))
    print(f"Written: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build_manual(root / "manual.pdf")
