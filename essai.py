class TreeNode:
    def __init__(self, label, content=None):
        self.label = label
        self.content = content
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)


def build_resume_tree(resume_data):
    root = TreeNode("Resume")
    
    contact_info = TreeNode("Contact Information", content=resume_data.get("contact_info", {}))
    root.add_child(contact_info)
    
    education = TreeNode("Education", content=resume_data.get("education", []))
    root.add_child(education)
    
    work_experience = TreeNode("Work Experience", content=resume_data.get("work_experience", []))
    root.add_child(work_experience)
    
    skills = TreeNode("Skills", content=resume_data.get("skills", []))
    root.add_child(skills)
    
    projects = TreeNode("Projects", content=resume_data.get("projects", []))
    root.add_child(projects)
    
    return root


# Example usage:
resume_data = {
    "contact_info": {
        "name": "John Doe",
        "email": "johndoe@email.com",
        "phone": "123-456-7890",
        "address": "123 Main St, City, State",
    },
    "education": [
        {
            "degree": "Bachelor of Science",
            "major": "Computer Science",
            "university": "XYZ University",
            "graduation_year": "2022",
        },
    ],
    "work_experience": [
        {
            "title": "Software Engineer",
            "company": "ABC Corp",
            "duration": "2022 - Present",
            "description": "Developed and maintained web applications.",
        },
    ],
    "skills": ["Python", "JavaScript", "Machine Learning"],
    "projects": [
        {
            "name": "Resume Parser",
            "description": "A Python tool to parse resumes and build semantic trees.",
        },
    ],
}

resume_tree = build_resume_tree(resume_data)

# You can then traverse and analyze the resume tree as needed.
print(resume_tree)