import xml.etree.ElementTree as ET
import pandas as pd

# Load the XML files
phenotype_tree = ET.parse("orphadata_files/en_product4.xml") 
functional_consequences_tree = ET.parse("orphadata_files/en_funct_consequences.xml")
natural_history_tree = ET.parse("orphadata_files/en_product9_ages.xml")
genes_tree = ET.parse("orphadata_files/en_product6.xml")
prevalence_tree = ET.parse("orphadata_files/en_product9_prev.xml")

phenotype_root = phenotype_tree.getroot()
functional_consequences_root = functional_consequences_tree.getroot()
natural_history_root = natural_history_tree.getroot()
genes_root = genes_tree.getroot()
prevalence_root = prevalence_tree.getroot()

# Build initial dictionary with phenotype data
diseases = {}

# Loop through phenotype XML for rare diseases
for disorder in phenotype_root.findall(".//Disorder"):
    orpha_code = disorder.find("OrphaCode").text
    name_elem = disorder.find(".//Name[@lang='en']")
    name = name_elem.text if name_elem is not None else "N/A"

    expert_link_elem = disorder.find(".//ExpertLink[@lang='en']")
    expert_link = expert_link_elem.text if expert_link_elem is not None else "N/A"

    # Extract additional fields from phenotype data
    disorder_type_elem = disorder.find(".//DisorderType/Name[@lang='en']")
    disorder_type = disorder_type_elem.text if disorder_type_elem is not None else "N/A"

    disorder_group_elem = disorder.find(".//DisorderGroup/Name[@lang='en']")
    disorder_group = disorder_group_elem.text if disorder_group_elem is not None else "N/A"

    # Initialize the disease entry (including placeholder for NaturalHistory)
    if orpha_code not in diseases:
        diseases[orpha_code] = {
            "OrphaCode": orpha_code,
            "Name": name,
            "ExpertLink": expert_link,
            "DisorderType": disorder_type,
            "DisorderGroup": disorder_group,
            "HPO_ID": [],
            "HPO_Term": [],
            "Frequency": [],
        }

    # Extract HPO (phenotype) associations for each disorder
    for assoc in disorder.findall(".//HPODisorderAssociation"):
        hpo_id_elem = assoc.find(".//HPOId")
        hpo_id = hpo_id_elem.text if hpo_id_elem is not None else "NA"
        hpo_term_elem = assoc.find(".//HPOTerm")
        hpo_term = hpo_term_elem.text if hpo_term_elem is not None else "NA"
        frequency_elem = assoc.find(".//HPOFrequency/Name[@lang='en']")
        frequency = frequency_elem.text if frequency_elem is not None else "NA"

        diseases[orpha_code]["HPO_ID"].append(hpo_id)
        diseases[orpha_code]["HPO_Term"].append(hpo_term)
        diseases[orpha_code]["Frequency"].append(frequency)

# Create a single DataFrame from the aggregated disease data and export to CSV
df = pd.DataFrame(list(diseases.values()))
df.to_csv("csv/phenotype_data.csv", index=False)

diseases = {}

# Loop through each Disorder in the natural history XML file
for disorder in natural_history_root.findall(".//Disorder"):
    # Get the OrphaCode and other metadata
    orpha_code = disorder.find("OrphaCode").text
    name_elem = disorder.find(".//Name[@lang='en']")
    name = name_elem.text if name_elem is not None else "NA"
    
    expert_link_elem = disorder.find(".//ExpertLink[@lang='en']")
    expert_link = expert_link_elem.text if expert_link_elem is not None else "NA"
    
    disorder_type_elem = disorder.find(".//DisorderType/Name[@lang='en']")
    disorder_type = disorder_type_elem.text if disorder_type_elem is not None else "NA"
    
    disorder_group_elem = disorder.find(".//DisorderGroup/Name[@lang='en']")
    disorder_group = disorder_group_elem.text if disorder_group_elem is not None else "NA"
    
    # Extract all the age-of-onset values by looping over each <AverageAgeOfOnset>
    onset_ages = []
    for elem in disorder.findall(".//AverageAgeOfOnset"):
        # Each AverageAgeOfOnset tag should contain a <Name lang="en"> tag with the descriptor
        name_elem = elem.find("Name[@lang='en']")
        if name_elem is not None:
            onset_ages.append(name_elem.text)
    type_inheritances = []
    for elem in disorder.findall(".//TypeOfInheritance"):
        name_elem = elem.find("Name[@lang='en']")
        if name_elem is not None:
            type_inheritances.append(name_elem.text)
            
    # Create or update the dictionary entry;
    # here we store the list of onset ages (or ["N/A"] in case no data are found)
    diseases[orpha_code] = {
         "OrphaCode": orpha_code,
         "Name": name,
         "AgeOfOnset": onset_ages if onset_ages else ["NA"],
         "TypeOfInheritance": type_inheritances if type_inheritances else ["NA"]
         
    }

# Convert the dictionary to a DataFrame and export to CSV or process further as needed.
df = pd.DataFrame(list(diseases.values()))
df.to_csv("csv/natural_history_data.csv", index=False)

diseases = {}

# Loop through each Disorder in the natural history XML file
for disorder in functional_consequences_root.findall(".//Disorder"):
    # Get the OrphaCode and other metadata
    orpha_code = disorder.find("OrphaCode").text
    name_elem = disorder.find(".//Name[@lang='en']")
    name = name_elem.text if name_elem is not None else "NA"
    
    expert_link_elem = disorder.find(".//ExpertLink[@lang='en']")
    expert_link = expert_link_elem.text if expert_link_elem is not None else "NA"
    
    disorder_type_elem = disorder.find(".//DisorderType/Name[@lang='en']")
    disorder_type = disorder_type_elem.text if disorder_type_elem is not None else "NA"
    
    disorder_group_elem = disorder.find(".//DisorderGroup/Name[@lang='en']")
    disorder_group = disorder_group_elem.text if disorder_group_elem is not None else "NA"
    
    functional_consequences = []
    loss_of_ability = []
    disability_types = []
    defined_statuses = []
    
    for elem in disorder.findall(".//DisabilityDisorderAssociation"):
        disability_elem = elem.find(".//Disability/Name[@lang='en']")
        if disability_elem is not None:
            functional_consequences.append(disability_elem.text)
            
        loss = elem.find("LossOfAbility")
        loss_of_ability.append(loss.text if loss is not None else "NA")
        
        type_elem = elem.find("Type") 
        disability_types.append(type_elem.text if type_elem is not None else "NA")
        
        defined = elem.find("Defined")
        defined_statuses.append(defined.text if defined is not None else "NA")
            
    # Create or update the dictionary entry
    diseases[orpha_code] = {
         "OrphaCode": orpha_code,
         "Name": name,
         "FunctionalConsequence": functional_consequences if functional_consequences else ["NA"],
         "LossOfAbility": loss_of_ability if loss_of_ability else ["NA"],
         "DisabilityType": disability_types if disability_types else ["NA"], 
         "Defined": defined_statuses if defined_statuses else ["NA"]
    }

# Convert the dictionary to a DataFrame and export to CSV or process further as needed.
df = pd.DataFrame(list(diseases.values()))
df.to_csv("csv/functional_consequences_data.csv", index=False)

diseases = {}

# Loop through each Disorder in the natural history XML file
for disorder in genes_root.findall(".//Disorder"):
    # Get the OrphaCode and other metadata
    orpha_code = disorder.find("OrphaCode").text
    name_elem = disorder.find(".//Name[@lang='en']")
    name = name_elem.text if name_elem is not None else "NA"
    
    expert_link_elem = disorder.find(".//ExpertLink[@lang='en']")
    expert_link = expert_link_elem.text if expert_link_elem is not None else "NA"
    
    disorder_type_elem = disorder.find(".//DisorderType/Name[@lang='en']")
    disorder_type = disorder_type_elem.text if disorder_type_elem is not None else "NA"
    
    disorder_group_elem = disorder.find(".//DisorderGroup/Name[@lang='en']")
    disorder_group = disorder_group_elem.text if disorder_group_elem is not None else "NA"
    
    genes = []
    
    for gene_assoc in disorder.findall(".//DisorderGeneAssociation"):
        gene_info = {}
        gene = gene_assoc.find("Gene")
        if gene is not None:
            # Basic gene info
            gene_info["id"] = gene.get("id")
            name_elem = gene.find("Name[@lang='en']")
            gene_info["name"] = name_elem.text if name_elem is not None else "NA"
            symbol_elem = gene.find("Symbol")
            gene_info["symbol"] = symbol_elem.text if symbol_elem is not None else "NA"
            
            # Synonyms
            synonyms = []
            for syn in gene.findall(".//Synonym[@lang='en']"):
                synonyms.append(syn.text)
            gene_info["synonyms"] = synonyms if synonyms else ["NA"]
            
            # Gene type
            gene_type = gene.find(".//GeneType/Name[@lang='en']")
            gene_info["type"] = gene_type.text if gene_type is not None else "NA"
            
            # External references
            ext_refs = {}
            for ref in gene.findall(".//ExternalReference"):
                source = ref.find("Source").text
                reference = ref.find("Reference").text
                ext_refs[source] = reference
            gene_info["external_references"] = ext_refs if ext_refs else {"NA": "NA"}
            
            # Locus information
            locus_list = []
            for locus in gene.findall(".//Locus"):
                locus_info = {
                    "id": locus.get("id"),
                    "gene_locus": locus.find("GeneLocus").text if locus.find("GeneLocus") is not None else "NA",
                    "locus_key": locus.find("LocusKey").text if locus.find("LocusKey") is not None else "NA"
                }
                locus_list.append(locus_info)
            gene_info["locus"] = locus_list if locus_list else [{"id": "NA", "gene_locus": "NA", "locus_key": "NA"}]
            
            # Association type and status
            assoc_type = gene_assoc.find(".//DisorderGeneAssociationType/Name[@lang='en']")
            gene_info["association_type"] = assoc_type.text if assoc_type is not None else "NA"
            
            assoc_status = gene_assoc.find(".//DisorderGeneAssociationStatus/Name[@lang='en']")
            gene_info["association_status"] = assoc_status.text if assoc_status is not None else "NA"
            
            genes.append(gene_info)
            
    # Create or update the dictionary entry
    diseases[orpha_code] = {
         "OrphaCode": orpha_code,
         "Name": name,
         "Genes": genes if genes else [{"NA": "NA"}]
    }

# Convert the dictionary to a DataFrame and export to CSV or process further as needed.
df = pd.DataFrame(list(diseases.values()))
df.to_csv("csv/genes_data.csv", index=False)

# Loop through each Disorder in the prevalence XML file to extract prevalence data
prevalence_data = {}

for disorder in prevalence_root.findall(".//Disorder"):
    # Get basic disorder info
    orpha_code = disorder.find("OrphaCode").text
    name_elem = disorder.find(".//Name[@lang='en']")
    name = name_elem.text if name_elem is not None else "NA"
    
    prevalences = []
    # Find the PrevalenceList within the disorder (it may contain one or more <Prevalence> records)
    prevalence_list = disorder.find(".//PrevalenceList")
    if prevalence_list is not None:
        for prev in prevalence_list.findall("Prevalence"):
            prev_info = {}
            # Extract the prevalence ID from the attribute
            prev_info["Prevalence_ID"] = prev.get("id", "NA")
            
            # Extract the Source of information
            source = prev.find("Source")
            prev_info["Source"] = source.text if source is not None else "NA"
            
            # Extract the Prevalence Type (e.g., Cases/families, Point prevalence)
            prevalence_type = prev.find("PrevalenceType/Name[@lang='en']")
            prev_info["PrevalenceType"] = prevalence_type.text if prevalence_type is not None else "NA"
            
            # Extract the Prevalence Qualification (e.g., Case(s), Class only)
            prevalence_qual = prev.find("PrevalenceQualification/Name[@lang='en']")
            prev_info["PrevalenceQualification"] = (
                prevalence_qual.text if prevalence_qual is not None else "NA"
            )
            
            # Extract the Prevalence Class; note that this element might be empty
            prevalence_class = prev.find("PrevalenceClass/Name[@lang='en']")
            prev_info["PrevalenceClass"] = (
                prevalence_class.text if prevalence_class is not None and prevalence_class.text is not None else "NA"
            )
            
            # Extract the mean value (ValMoy) and convert it as a float when possible
            val_moy = prev.find("ValMoy")
            prev_info["ValMoy"] = float(val_moy.text) if val_moy is not None and val_moy.text is not None else None
            
            # Extract the Prevalence Geographic information
            prevalence_geo = prev.find("PrevalenceGeographic/Name[@lang='en']")
            prev_info["PrevalenceGeographic"] = (
                prevalence_geo.text if prevalence_geo is not None else "NA"
            )
            
            # Extract the Prevalence Validation Status
            prevalence_valid = prev.find("PrevalenceValidationStatus/Name[@lang='en']")
            prev_info["PrevalenceValidationStatus"] = (
                prevalence_valid.text if prevalence_valid is not None else "NA"
            )
            
            # Add this prevalence record to the list for the disorder
            prevalences.append(prev_info)
    
    # Save the disorder's prevalence data (if no records found, use a default placeholder)
    prevalence_data[orpha_code] = {
        "OrphaCode": orpha_code,
        "Name": name,
        "Prevalences": prevalences if prevalences else [{"NA": "NA"}]
    }

# Convert the dictionary to a Pandas DataFrame and export in CSV format
df_prevalence = pd.DataFrame(list(prevalence_data.values()))
df_prevalence.to_csv("csv/prevalence_data.csv", index=False)