# Contributing to automation

When contributing to this automation, please first discuss the changes you wish to make.

## How to use:
1. Clone the [repository](https://github.com/HabanaAI/Gaudi-tutorials.git) to your machine.


	```bash
	git clone https://github.com/HabanaAI/Gaudi-tutorials.git
	cd Gaudi-tutorials/Pytorch/automation
	```

	**do not** work on the `main` branch.

2. Create a new branch to hold your changes:
	```bash
	git checkout -b a-descriptive-name-for-your-changes
	```

3.  Once you're happy with your changes, add the changed files using `git add` and make a commit with `git commit` to record your changes locally:

	```bash
	git add modified_file
	git commit
	```

4.	It is a good idea to sync your copy of the code with the original
	repository regularly. This way you can quickly account for changes:

	```bash
	git fetch upstream
	git rebase upstream/main
    ```

5.   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project for review.
