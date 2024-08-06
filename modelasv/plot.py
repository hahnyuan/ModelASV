import matplotlib.pyplot as plt


def simulation_reports_compare(
    report1,
    report2,
    model_name,
    report1_name="report1",
    report2_name="report2",
    key="inference_time",
    verbose=False,
    fig_size=(15, 6),
):
    value_count1 = {}
    value_count2 = {}
    for name in report1:
        r1 = report1[name]
        r2 = report2[name]
        r2_time = r2[0][key]
        r1_time = r1[0][key]
        if verbose:
            print(name)
            print(r1)
            print(r2)
            print(r2_time / r1_time, r2_time, r1_time)
        typename = name.split(".")[-1]
        if typename not in value_count1:
            value_count1[typename] = 0
            value_count2[typename] = 0
        value_count2[typename] += r2_time
        value_count1[typename] += r1_time

    fig = plt.figure(figsize=fig_size)

    # Extract the typename and value values from value_count1 and value_count2 dictionaries
    typenames = list(value_count1.keys())
    typenames = [_.replace("_", "\n") for _ in typenames]
    value1 = list(value_count1.values())
    value2 = list(value_count2.values())

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = range(len(typenames))
    r2 = [x + bar_width for x in r1]

    # Plot the bar chart
    plt.bar(r1, value1, color="blue", width=bar_width, label=report1_name)
    plt.bar(r2, value2, color="orange", width=bar_width, label=report2_name)

    # Add labels and title
    plt.xlabel("Layer")
    plt.ylabel(key)
    plt.title(f"Comparison of {key} Count ({model_name})")

    # Add x-axis tick labels
    plt.xticks([r + bar_width / 2 for r in range(len(typenames))], typenames)

    # Add legend
    plt.legend()
