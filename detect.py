import tkinter as tk
import pandas as pd

class CriminalInfoApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Criminal Information Form")

        # Define the path to the Excel file
        self.excel_file_path = "C:/Users/samay/Downloads/Face-recognition-Using-Facenet-On-Tensorflow-2.X-master/extracted/Face-recognition-Using-Facenet-On-Tensorflow-2.X-master/criminal data.xlsx"

        # Create and place the labels and entry widgets
        self.create_widgets()

        # Fetch and populate data from the Excel file
        self.data = self.fetch_data_from_excel()
        self.populate_form(self.data)

    def create_widgets(self):
        tk.Label(self.root, text="Criminal Name:").grid(row=0, column=0, padx=10, pady=5)
        self.entry_name = tk.Entry(self.root)
        self.entry_name.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Criminal ID:").grid(row=1, column=0, padx=10, pady=5)
        self.entry_id = tk.Entry(self.root)
        self.entry_id.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Crime Type:").grid(row=2, column=0, padx=10, pady=5)
        self.entry_crime_type = tk.Entry(self.root)
        self.entry_crime_type.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Age:").grid(row=3, column=0, padx=10, pady=5)
        self.entry_age = tk.Entry(self.root)
        self.entry_age.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Height (m):").grid(row=4, column=0, padx=10, pady=5)
        self.entry_height = tk.Entry(self.root)
        self.entry_height.grid(row=4, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Weight (kg):").grid(row=5, column=0, padx=10, pady=5)
        self.entry_weight = tk.Entry(self.root)
        self.entry_weight.grid(row=5, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Occupation:").grid(row=6, column=0, padx=10, pady=5)
        self.entry_occupation = tk.Entry(self.root)
        self.entry_occupation.grid(row=6, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Address:").grid(row=7, column=0, padx=10, pady=5)
        self.entry_address = tk.Entry(self.root)
        self.entry_address.grid(row=7, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Arrest Type:").grid(row=8, column=0, padx=10, pady=5)
        self.entry_arrest_type = tk.Entry(self.root)
        self.entry_arrest_type.grid(row=8, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Gender:").grid(row=9, column=0, padx=10, pady=5)
        self.entry_gender = tk.Entry(self.root)
        self.entry_gender.grid(row=9, column=1, padx=10, pady=5)

        # Create and place the label to display the result
        self.display_label = tk.Label(self.root, text="", justify="left")
        self.display_label.grid(row=10, columnspan=2, pady=10)

    def fetch_data_from_excel(self):
        # Read the Excel file
        try:
            df = pd.read_excel(self.excel_file_path)
            # Assuming the data is in the first row and columns are named appropriately
            if not df.empty:
                data = df.iloc[0]
                return {
                    'name': data.get('Criminal Name', ''),
                    'criminal_id': data.get('Criminal ID', ''),
                    'crime_type': data.get('Crime Type', ''),
                    'age': data.get('Age', ''),
                    'height': data.get('Height (m)', ''),
                    'weight': data.get('Weight (kg)', ''),
                    'occupation': data.get('Occupation', ''),
                    'address': data.get('Address', ''),
                    'arrest_type': data.get('Arrest Type', ''),
                    'gender': data.get('Gender', '')
                }
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return {}

    def populate_form(self, data):
        # Populate the form with data from the Excel file
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, data.get('name', ''))

        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, data.get('criminal_id', ''))

        self.entry_crime_type.delete(0, tk.END)
        self.entry_crime_type.insert(0, data.get('crime_type', ''))

        self.entry_age.delete(0, tk.END)
        self.entry_age.insert(0, data.get('age', ''))

        self.entry_height.delete(0, tk.END)
        self.entry_height.insert(0, data.get('height', ''))

        self.entry_weight.delete(0, tk.END)
        self.entry_weight.insert(0, data.get('weight', ''))

        self.entry_occupation.delete(0, tk.END)
        self.entry_occupation.insert(0, data.get('occupation', ''))

        self.entry_address.delete(0, tk.END)
        self.entry_address.insert(0, data.get('address', ''))

        self.entry_arrest_type.delete(0, tk.END)
        self.entry_arrest_type.insert(0, data.get('arrest_type', ''))

        self.entry_gender.delete(0, tk.END)
        self.entry_gender.insert(0, data.get('gender', ''))


# Create the Tkinter window and run the application
name = "main"
if  name == "_main_":
    root = tk.Tk()
    app = CriminalInfoApp(root)
    root.mainloop()