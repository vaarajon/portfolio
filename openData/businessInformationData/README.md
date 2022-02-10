# Business Information Data using the open API provided by the Finnish Patent and Registration Office (PRH)

## Summary
Simple application which fetch the data for the companies that are defined
in "yritykset.csv" -file.

## What
1. The application will fetch data for the companies that are defined
in yritykset.csv -file.
2. Flatten the json data into a dataframe
3. Append the flattened data into a PRH_data dataframe
4. Convert the PRH_data dataframe into a csv-file and save it to the workspace folder

## Installation
1. Clone the repository to your local machine 
`$ git clone https://github.com/vaarajon/portfolio.git`

2. Browse to the directory: `portfolio/openData/businessInformationData`

3. Make sure you have needed packages installed in your virtual environment
- pandas

## How does it work
1. Add name of the companies to the csv-file
2. Run the code
3. Open the dataframe or the PRH_data.csv -file

